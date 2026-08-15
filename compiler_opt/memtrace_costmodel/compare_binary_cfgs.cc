#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "compiler_opt/memtrace_costmodel/compare_binary_cfgs_lib.h"
#include "nlohmann/json.hpp"

ABSL_FLAG(std::string, binary_path_a, "", "The path to binary A.");
ABSL_FLAG(std::string, corpus_path_a, "", "The path to corpus A.");
ABSL_FLAG(std::string, binary_path_b, "", "The path to binary B.");
ABSL_FLAG(std::string, corpus_path_b, "", "The path to corpus B.");
ABSL_FLAG(bool, continue_on_diff, false,
          "Whether or not to continue running if a difference is found.");

absl::flat_hash_map<std::string,
                    const std::vector<mlgo::latency_model::ControlFlowEdge>>
GetCfgFromBinaryOrCorpus(absl::string_view binary_path,
                         absl::string_view corpus_path) {
  if (!binary_path.empty()) {
    absl::StatusOr<absl::flat_hash_map<
        std::string, const std::vector<mlgo::latency_model::ControlFlowEdge>>>
        binary_cfgs = mlgo::latency_model::LoadBinaryCfgs(binary_path);
    QCHECK_OK(binary_cfgs);
    return *binary_cfgs;
  }

  std::vector<std::string> module_full_paths;

  std::string corpus_json_path =
      (std::filesystem::path(std::string(corpus_path)) /
       "corpus_description.json")
          .string();
  std::ifstream corpus_file(corpus_json_path);
  QCHECK(corpus_file) << "Failed to open corpus JSON path: "
                      << corpus_json_path;
  std::stringstream buffer;
  buffer << corpus_file.rdbuf();
  std::string corpus_json_contents = buffer.str();

  nlohmann::json corpus_description =
      nlohmann::json::parse(corpus_json_contents);

  QCHECK(corpus_description.contains("modules"));
  QCHECK(corpus_description["modules"].is_array());

  for (const std::string module_path : corpus_description["modules"]) {
    std::string module_full_path =
        (std::filesystem::path(std::string(corpus_path)) / module_path)
            .string() +
        ".bc.o";
    module_full_paths.push_back(std::move(module_full_path));
  }

  absl::StatusOr<absl::flat_hash_map<
      std::string, const std::vector<mlgo::latency_model::ControlFlowEdge>>>
      corpus_cfgs = mlgo::latency_model::LoadCorpusCfgs(module_full_paths);
  QCHECK_OK(corpus_cfgs);

  return *corpus_cfgs;
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  if (absl::GetFlag(FLAGS_binary_path_a).empty() &&
      absl::GetFlag(FLAGS_corpus_path_a).empty()) {
    LOG(QFATAL) << "--binary_path_a or --corpus_path_a needs to be specified.";
  }

  if (!absl::GetFlag(FLAGS_binary_path_a).empty() &&
      !absl::GetFlag(FLAGS_corpus_path_a).empty()) {
    LOG(QFATAL) << "--binary_path_a and --corpus_path_a cannot both be set at "
                   "the same time.";
  }

  if (absl::GetFlag(FLAGS_binary_path_b).empty() &&
      absl::GetFlag(FLAGS_corpus_path_b).empty()) {
    LOG(QFATAL) << "--binary_path_b or --corpus_path_b needs to be specified.";
  }

  if (!absl::GetFlag(FLAGS_binary_path_b).empty() &&
      !absl::GetFlag(FLAGS_corpus_path_b).empty()) {
    LOG(QFATAL) << "--binary_path_b and --corpus_path_b cannot both be set at "
                   "the same time.";
  }

  absl::flat_hash_map<std::string,
                      const std::vector<mlgo::latency_model::ControlFlowEdge>>
      a_cfgs = GetCfgFromBinaryOrCorpus(absl::GetFlag(FLAGS_binary_path_a),
                                        absl::GetFlag(FLAGS_corpus_path_a));
  absl::flat_hash_map<std::string,
                      const std::vector<mlgo::latency_model::ControlFlowEdge>>
      b_cfgs = GetCfgFromBinaryOrCorpus(absl::GetFlag(FLAGS_binary_path_b),
                                        absl::GetFlag(FLAGS_corpus_path_b));

  bool continue_on_diff = absl::GetFlag(FLAGS_continue_on_diff);

  if (a_cfgs.size() != b_cfgs.size()) {
    std::cout << "Binaries have a different function set.\n";
    if (!continue_on_diff) return 0;
  }

  for (const auto& function_info : a_cfgs) {
    const auto function_b_info_it = b_cfgs.find(function_info.first);
    if (function_b_info_it == b_cfgs.end()) {
      std::cout << "Failed to find function " << function_info.first
                << " from binary A in binary B.\n";
      if (!continue_on_diff) return 0;
    }

    bool are_cfgs_different = mlgo::latency_model::AreCfgsDifferent(
        function_info.second, function_b_info_it->second);
    if (are_cfgs_different) {
      std::cout << "Function " << function_info.first
                << " has a different CFG between the two versions.\n";
      if (!continue_on_diff) return 0;
    }
  }

  return 0;
}
