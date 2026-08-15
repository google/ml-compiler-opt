#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "compiler_opt/memtrace_costmodel/basic_block_trace.h"
#include "compiler_opt/memtrace_costmodel/mbb_trace.proto.h"
#include "compiler_opt/memtrace_costmodel/serialize_trace_functions_lib.h"
#include "compiler_opt/memtrace_costmodel/serialized_mbbs.proto.h"
#include "llvm-c/Target.h"
#include "nlohmann/json.hpp"
#include "riegeli/bytes/file_reader.h"
#include "riegeli/bytes/file_writer.h"
#include "riegeli/records/record_reader.h"
#include "riegeli/records/record_writer.h"

ABSL_FLAG(std::string, trace_path, "", "Path to the trace recordio.");
ABSL_FLAG(std::string, function_index_path, "",
          "The path to the function index.");
ABSL_FLAG(std::string, output_path, "", "The path to the output proto.");
ABSL_FLAG(std::string, exclude_functions_list, "",
          "The path to the file listing function names to exclude.");
ABSL_FLAG(std::string, corpus_path, "",
          "The path to the corpus description JSON to pull blocks from.");
ABSL_FLAG(std::string, target_triple, "x86_64", "The target triple to use.");

namespace {
std::vector<std::string> GetObjectPaths(absl::string_view corpus_path) {
  std::string corpus_path_str(corpus_path);
  std::ifstream corpus_file(corpus_path_str);
  QCHECK(corpus_file) << "Failed to open corpus path: " << corpus_path;
  std::stringstream buffer;
  buffer << corpus_file.rdbuf();
  std::string corpus_description_contents = buffer.str();

  nlohmann::json corpus_description =
      nlohmann::json::parse(corpus_description_contents);

  auto modules_it = corpus_description.find("modules");
  CHECK(modules_it != corpus_description.end());

  std::vector<std::string> object_file_paths;
  object_file_paths.reserve(modules_it->size());

  std::string corpus_dir_path =
      std::filesystem::path(std::string(corpus_path)).parent_path().string();

  for (const std::string relative_module_path : *modules_it) {
    object_file_paths.push_back(
        (std::filesystem::path(corpus_dir_path) / relative_module_path)
            .string() +
        ".bc.o");
  }

  return object_file_paths;
}
}  // namespace

int main(int argc, char** argv) {
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86Disassembler();

  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  const std::string trace_path = absl::GetFlag(FLAGS_trace_path);
  const std::string function_index_path =
      absl::GetFlag(FLAGS_function_index_path);
  const std::string output_path = absl::GetFlag(FLAGS_output_path);
  const std::string exclude_functions_list =
      absl::GetFlag(FLAGS_exclude_functions_list);
  const std::string corpus_path = absl::GetFlag(FLAGS_corpus_path);

  if (trace_path.empty()) {
    LOG(QFATAL) << "--trace_path was not specified.";
  }

  if (function_index_path.empty()) {
    LOG(QFATAL) << "--function_index_path was not specified.";
  }

  if (output_path.empty()) {
    LOG(QFATAL) << "--output_path was not specified.";
  }

  if (exclude_functions_list.empty()) {
    LOG(QFATAL) << "--exclude_functions_list was not specified.";
  }

  if (corpus_path.empty()) {
    LOG(QFATAL) << "--corpus_path was not specified.";
  }

  std::vector<std::string> excluded_functions =
      mlgo::latency_model::LoadExcludedFunctions(exclude_functions_list);

  riegeli::RecordReader function_index_reader(
      riegeli::Maker<riegeli::FileReader>(function_index_path));
  mlgo::latency_model::FunctionMapping function_name_to_id;
  CHECK(function_index_reader.ReadRecord(function_name_to_id));
  QCHECK(function_index_reader.Close()) << function_index_reader.status();

  std::vector<mlgo::latency_model::MbbTrace> mbb_traces;
  riegeli::RecordReader trace_reader(
      riegeli::Maker<riegeli::FileReader>(trace_path));
  mlgo::latency_model::MbbTrace current_mbb_trace;
  while (trace_reader.ReadRecord(current_mbb_trace)) {
    mbb_traces.push_back(std::move(current_mbb_trace));
  }
  QCHECK(trace_reader.Close()) << trace_reader.status();

  absl::flat_hash_set<mlgo::latency_model::MachineBbId,
                      mlgo::latency_model::MachineBBIDKeyHash,
                      mlgo::latency_model::MachineBBIDKeyEqual>
      required_mbbs = mlgo::latency_model::GetRequiredMbbs(
          mbb_traces, excluded_functions, function_name_to_id);

  std::vector<std::string> object_file_paths = GetObjectPaths(corpus_path);

  mlgo::latency_model::SerializedMbbs serialized_mbbs =
      mlgo::latency_model::GetSerializedMbbs(
          object_file_paths, absl::GetFlag(FLAGS_target_triple),
          function_name_to_id, required_mbbs);

  riegeli::RecordWriter output_writer(
      riegeli::Maker<riegeli::FileWriter>(output_path),
      riegeli::RecordWriterBase::Options().set_zstd());
  output_writer.WriteRecord(serialized_mbbs);
  QCHECK(output_writer.Close()) << output_writer.status();

  return 0;
}
