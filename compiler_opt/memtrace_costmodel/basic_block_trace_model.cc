#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/flags/parse.h"
#include "compiler_opt/memtrace_costmodel/basic_block_trace.h"
#undef X86
#undef X86_64
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "compiler_opt/memtrace_costmodel/costmodel.h"
#include "compiler_opt/memtrace_costmodel/costmodel_factory.h"
#include "compiler_opt/memtrace_costmodel/mbb_trace.proto.h"
#include "compiler_opt/memtrace_costmodel/serialized_mbbs.proto.h"
#include "llvm-c/Target.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/lib/Target/X86/MCTargetDesc/X86MCTargetDesc.h"
#include "nlohmann/json.hpp"
#include "riegeli/bytes/file_reader.h"
#include "riegeli/records/record_reader.h"

ABSL_FLAG(std::string, bb_trace_path, "", "The path to the basic block trace.");
ABSL_FLAG(std::string, binary_path, "", "The path to the binary.");
ABSL_FLAG(std::string, corpus_path, "",
          "The path to the corpus description JSON.");
ABSL_FLAG(std::string, target_triple, "x86_64",
          "The target triple of the binary.");
ABSL_FLAG(std::string, cpu_name, "skylake", "The CPU name to model.");
ABSL_FLAG(std::string, function_index_path, "",
          "The path to the function name to ID mapping.");
ABSL_FLAG(std::string, serialized_bbs_path, "",
          "The path to the serialized basic blocks to load.");

// Model specific flags.
ABSL_FLAG(mlgo::latency_model::CostModelType, model_type,
          mlgo::latency_model::CostModelType::Mca,
          "The type of cost model to use. (\"mca\" | \"print\" "
          "| \"instruction_counting\")");
ABSL_FLAG(std::string, print_output_file, "",
          "The output file if the print cost model is selected.");

int main(int argc, char** argv) {
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86Disassembler();
  LLVMInitializeX86TargetMCA();

  absl::ParseCommandLine(argc, argv);

  if (absl::GetFlag(FLAGS_bb_trace_path).empty()) {
    LOG(QFATAL) << "--bb_trace_path was not specified.";
  }

  if (absl::GetFlag(FLAGS_binary_path).empty() &&
      absl::GetFlag(FLAGS_corpus_path).empty()) {
    LOG(QFATAL) << "--binary_path or --corpus_path needs to be specified.";
  }

  if (!absl::GetFlag(FLAGS_binary_path).empty() &&
      !absl::GetFlag(FLAGS_corpus_path).empty()) {
    LOG(QFATAL) << "--binary_path and --corpus_path cannot both be specified "
                   "at the same "
                   "time.";
  }

  if (absl::GetFlag(FLAGS_function_index_path).empty()) {
    LOG(QFATAL) << "--function_index_path was not specified.";
  }

  mlgo::latency_model::ValidateCostModelFlags(
      absl::GetFlag(FLAGS_model_type), "",
      absl::GetFlag(FLAGS_print_output_file));

  riegeli::RecordReader function_index_reader(
      riegeli::Maker<riegeli::FileReader>(
          absl::GetFlag(FLAGS_function_index_path)));
  mlgo::latency_model::FunctionMapping function_name_to_id;
  QCHECK(function_index_reader.ReadRecord(function_name_to_id));
  QCHECK(function_index_reader.Close()) << function_index_reader.status();

  std::unique_ptr<mlgo::latency_model::ApplicationToBbDisassembler>
      application_disassembler;
  if (!absl::GetFlag(FLAGS_binary_path).empty()) {
    application_disassembler = std::make_unique<
        mlgo::latency_model::BinaryApplicationToBbDisassembler>(
        absl::GetFlag(FLAGS_target_triple), absl::GetFlag(FLAGS_binary_path));
    application_disassembler->LoadBasicBlocks(function_name_to_id);
  } else {
    // Load module list from corpus JSON file using standard C++ streams.
    std::string corpus_description_path = absl::GetFlag(FLAGS_corpus_path);
    std::ifstream corpus_file(corpus_description_path);
    if (!corpus_file) {
      LOG(QFATAL) << "Failed to open corpus path: " << corpus_description_path;
    }
    std::stringstream buffer;
    buffer << corpus_file.rdbuf();
    std::string corpus_description_contents = buffer.str();
    nlohmann::json corpus_description =
        nlohmann::json::parse(corpus_description_contents);

    QCHECK(corpus_description.contains("modules"));
    QCHECK(corpus_description["modules"].is_array());

    std::vector<std::string> module_full_paths;
    module_full_paths.reserve(corpus_description["modules"].size());

    std::string corpus_path = absl::GetFlag(FLAGS_corpus_path);
    std::string corpus_dir_path =
        std::string(std::filesystem::path(corpus_path).parent_path());

    for (const std::string relative_module_path :
         corpus_description["modules"]) {
      module_full_paths.push_back(corpus_dir_path + "/" + relative_module_path +
                                  ".bc.o");
    }

    application_disassembler = std::make_unique<
        mlgo::latency_model::CorpusApplicationToBbDisassembler>(
        absl::GetFlag(FLAGS_target_triple), module_full_paths);
    application_disassembler->LoadBasicBlocks(function_name_to_id);
  }

  if (!absl::GetFlag(FLAGS_serialized_bbs_path).empty()) {
    riegeli::RecordReader serialized_bbs_reader(
        riegeli::Maker<riegeli::FileReader>(
            absl::GetFlag(FLAGS_serialized_bbs_path)));
    mlgo::latency_model::SerializedMbbs serialized_mbbs;
    QCHECK(serialized_bbs_reader.ReadRecord(serialized_mbbs));
    QCHECK(serialized_bbs_reader.Close()) << serialized_bbs_reader.status();

    application_disassembler->LoadSerializedBbs(serialized_mbbs);
  }

  const std::string& bb_trace_path = absl::GetFlag(FLAGS_bb_trace_path);

  riegeli::RecordReader trace_reader(
      riegeli::Maker<riegeli::FileReader>(bb_trace_path));

  std::string possible_lookup_error;
  llvm::Triple target_triple(absl::GetFlag(FLAGS_target_triple));
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(target_triple, possible_lookup_error);
  QCHECK_NE(target, nullptr) << possible_lookup_error;

  auto target_machine = std::unique_ptr<llvm::TargetMachine>(
      target->createTargetMachine(target_triple, absl::GetFlag(FLAGS_cpu_name),
                                  "", llvm::TargetOptions(), std::nullopt));

  auto cost_model_factory = mlgo::latency_model::CreateCostModelFactory(
      absl::GetFlag(FLAGS_model_type), target_machine.get(),
      absl::GetFlag(FLAGS_cpu_name), absl::GetFlag(FLAGS_target_triple), "", 0,
      absl::GetFlag(FLAGS_print_output_file));

  mlgo::latency_model::MbbTrace mbb_trace;
  while (trace_reader.ReadRecord(mbb_trace)) {
    application_disassembler->LoadSharedObjectTraces(mbb_trace);
    auto cost_model = cost_model_factory();

    for (const mlgo::latency_model::MachineBbId& basic_block :
         mbb_trace.mbbs()) {
      llvm::ArrayRef<mlgo::latency_model::InstructionInfo> bb_instructions =
          application_disassembler->GetDisassembledInstructions(basic_block);
      for (const mlgo::latency_model::InstructionInfo& instruction :
           bb_instructions) {
        cost_model->AddInstruction(instruction.instruction);
      }
    }

    std::cout << "Segment Cost: " << cost_model->GetCost() << "\n";
  }

  QCHECK(trace_reader.Close()) << trace_reader.status();
  return 0;
}
