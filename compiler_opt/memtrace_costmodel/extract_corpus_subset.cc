#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "compiler_opt/memtrace_costmodel/extract_corpus_subset_lib.h"
#include "compiler_opt/memtrace_costmodel/mbb_trace.proto.h"
#include "nlohmann/json.hpp"
#include "riegeli/bytes/file_reader.h"
#include "riegeli/records/record_reader.h"

ABSL_FLAG(std::string, corpus_json_path, "", "The corpus to process.");
ABSL_FLAG(std::string, bb_trace_path, "", "The path to the input trace.");
ABSL_FLAG(std::string, output_path, "",
          "The output path to put the corpus subset in.");
ABSL_FLAG(std::string, function_index_path, "",
          "The path to the function name to ID mapping.");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  if (absl::GetFlag(FLAGS_corpus_json_path).empty()) {
    LOG(QFATAL) << "--corpus_path must be set.\n";
  }

  if (absl::GetFlag(FLAGS_bb_trace_path).empty()) {
    LOG(QFATAL) << "--bb_trace_path must be set.\n";
  }

  if (absl::GetFlag(FLAGS_output_path).empty()) {
    LOG(QFATAL) << "--output_path must be set.\n";
  }

  if (absl::GetFlag(FLAGS_function_index_path).empty()) {
    LOG(QFATAL) << "--function_index_path was not specified.";
  }

  riegeli::RecordReader function_index_reader(
      riegeli::Maker<riegeli::FileReader>(
          absl::GetFlag(FLAGS_function_index_path)));
  mlgo::latency_model::FunctionMapping function_name_to_id;
  QCHECK(function_index_reader.ReadRecord(function_name_to_id));
  QCHECK(function_index_reader.Close()) << function_index_reader.status();

  std::string corpus_path = absl::GetFlag(FLAGS_corpus_json_path);

  std::ifstream corpus_file(corpus_path);
  QCHECK(corpus_file) << "Failed to open corpus JSON path: " << corpus_path;
  std::stringstream buffer;
  buffer << corpus_file.rdbuf();
  std::string corpus_json_contents = buffer.str();

  std::cout << "Loaded corpus JSON\n";

  nlohmann::json corpus_description =
      nlohmann::json::parse(corpus_json_contents);

  std::string corpus_base_path =
      std::filesystem::path(corpus_path).parent_path().string();

  // Check that the corpus has a modules field.
  QCHECK(corpus_description.contains("modules"));
  QCHECK(corpus_description["modules"].is_array());

  riegeli::RecordReader trace_reader(
      riegeli::Maker<riegeli::FileReader>(absl::GetFlag(FLAGS_bb_trace_path)));

  mlgo::latency_model::MbbTrace current_trace;
  absl::flat_hash_set<std::string> included_files;

  mlgo::latency_model::FunctionMapping processed_function_mapping =
      mlgo::latency_model::ProcessFunctionMappingForModifiedFunctions(
          function_name_to_id);

  absl::flat_hash_map<uint32_t, std::string> function_to_module_name =
      mlgo::latency_model::GetFunctionToModuleMapping(
          corpus_description["modules"], corpus_base_path,
          processed_function_mapping);

  while (trace_reader.ReadRecord(current_trace)) {
    absl::flat_hash_set<std::string> current_trace_files =
        mlgo::latency_model::GetIncludedFilesList(function_to_module_name,
                                                  current_trace);
    included_files.insert(current_trace_files.begin(),
                          current_trace_files.end());
  }
  QCHECK(trace_reader.Close()) << trace_reader.status();

  std::string output_corpus_base_path = absl::GetFlag(FLAGS_output_path);

  // The first BB in the trace should be from the entrypoint function.
  QCHECK_OK(mlgo::latency_model::CopySubsetCorpus(
      corpus_base_path, output_corpus_base_path, included_files,
      corpus_description));

  return 0;
}
