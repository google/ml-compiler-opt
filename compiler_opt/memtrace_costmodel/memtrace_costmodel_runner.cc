#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "compiler_opt/memtrace_costmodel/costmodel.h"
#include "compiler_opt/memtrace_costmodel/costmodel_factory.h"
#include "compiler_opt/memtrace_costmodel/memtrace_costmodel.h"
#undef X86
#undef X86_64
#include "llvm-c/Target.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Triple.h"

struct TargetTripleString {
  std::string target_triple = "";
};

std::string AbslUnparseFlag(TargetTripleString target_triple_string) {
  return target_triple_string.target_triple;
}

bool AbslParseFlag(absl::string_view text,
                   TargetTripleString* target_triple_string,
                   std::string* error) {
  const llvm::Target* const llvm_target = llvm::TargetRegistry::lookupTarget(
      llvm::Triple(llvm::StringRef(text)), *error);
  if (!llvm_target) return false;
  target_triple_string->target_triple = text;
  return true;
}

ABSL_FLAG(std::string, memtrace_path, "",
          "The path to the memtrace to process.");
// TODO: Lookup the symbol address and size automatically from
// a symbol name rather than requiring the user specify them manually.
ABSL_FLAG(uint64_t, symbol_address, 0, "The address of the entrypoint symbol");
ABSL_FLAG(TargetTripleString, target_triple, {"x86_64"},
          "The target triple of the platform to model");
ABSL_FLAG(std::string, llvm_cpu_name, "skylake",
          "The name of the CPU microarchitecture of the platform to model");
ABSL_FLAG(bool, split_on_segment, false,
          "Whether or not to split entrypoints by segment");

// Model specific flags.
ABSL_FLAG(mlgo::latency_model::CostModelType, model_type,
          mlgo::latency_model::CostModelType::Mca,
          "The type of cost model to use. (\"mca\" | \"print\" "
          "| \"instruction_counting\")");
ABSL_FLAG(std::string, print_output_file, "",
          "The output file if the print cost model is selected.");

int main(int argc, char** argv) {
#ifdef __x86_64__
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86Disassembler();
  LLVMInitializeX86TargetMCA();
#else
#error memtrace_costmodel_runner is only supported on X86_64.
#endif  // __x86_64__

  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  if (absl::GetFlag(FLAGS_memtrace_path).empty()) {
    LOG(ERROR) << "--memtrace_path was not specified.";
    return 1;
  }

  if (absl::GetFlag(FLAGS_symbol_address) == 0) {
    LOG(ERROR) << "--symbol_address must specify a symbol address.";
    return 1;
  }

  mlgo::latency_model::ValidateCostModelFlags(
      absl::GetFlag(FLAGS_model_type), "",
      absl::GetFlag(FLAGS_print_output_file));

  std::string possible_lookup_error;
  llvm::Triple triple(absl::GetFlag(FLAGS_target_triple).target_triple);
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(triple, possible_lookup_error);
  auto target_machine = std::unique_ptr<llvm::TargetMachine>(
      target->createTargetMachine(triple, absl::GetFlag(FLAGS_llvm_cpu_name),
                                  "", llvm::TargetOptions(), std::nullopt));

  std::function<std::unique_ptr<mlgo::latency_model::CostModel>()>
      cost_model_factory = mlgo::latency_model::CreateCostModelFactory(
          absl::GetFlag(FLAGS_model_type), target_machine.get(),
          absl::GetFlag(FLAGS_llvm_cpu_name),
          absl::GetFlag(FLAGS_target_triple).target_triple, "", 0,
          absl::GetFlag(FLAGS_print_output_file));

  QCHECK_OK(mlgo::latency_model::GetMemtraceCost(
      FLAGS_memtrace_path.CurrentValue(), {absl::GetFlag(FLAGS_symbol_address)},
      [](std::vector<double> entrypoint_segment_costs) {
        std::cout << "BEGIN ENTRYPOINT\n";
        for (const double entrypoint_segment_cost : entrypoint_segment_costs) {
          std::cout << entrypoint_segment_cost << "\n";
        }
        std::cout << "END ENTRYPOINT\n";
      },
      std::move(cost_model_factory), absl::GetFlag(FLAGS_split_on_segment),
      absl::GetFlag(FLAGS_target_triple).target_triple,
      absl::GetFlag(FLAGS_llvm_cpu_name)));

  LOG(INFO) << "Finished processing memtrace.";

  return 0;
}
