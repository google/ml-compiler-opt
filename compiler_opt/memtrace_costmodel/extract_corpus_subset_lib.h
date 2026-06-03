#ifndef COMPILER_OPT_MEMTRACE_COSTMODEL_EXTRACT_CORPUS_SUBSET_LIB_H_
#define COMPILER_OPT_MEMTRACE_COSTMODEL_EXTRACT_CORPUS_SUBSET_LIB_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "compiler_opt/memtrace_costmodel/mbb_trace.proto.h"
#include "nlohmann/json.hpp"

namespace mlgo {
namespace latency_model {

// Returns a mapping of functions to the modules that they are
// contained within.
absl::flat_hash_map<uint32_t, std::string> GetFunctionToModuleMapping(
    const std::vector<std::string>& module_paths, absl::string_view corpus_path,
    const FunctionMapping& function_name_to_id);

// Gets the minimal set of modules that need to be included in the corpus
// to completely cover all the functions in the MBB trace.
absl::flat_hash_set<std::string> GetIncludedFilesList(
    const absl::flat_hash_map<uint32_t, std::string>& function_to_module_names,
    const MbbTrace& mbb_trace);

// Copies the corpus subset containing the relevant modules in the trace
// to the output corpus path in addition to modifying the corpus
// description to fit the module set in the subset.
// TODO: We ideally want to avoid the JSON dependency here
// and pass in a better representation of the corpus description.
absl::Status CopySubsetCorpus(
    absl::string_view input_corpus_base_path,
    absl::string_view output_corpus_base_path,
    const absl::flat_hash_set<std::string>& included_modules,
    nlohmann::json& corpus_description);

// Processes a function mapping and adds additional entries pointing at
// existing function IDs to handle cases like function specialization
// and hot cold splitting that occur within PGO optimized binaries.
FunctionMapping ProcessFunctionMappingForModifiedFunctions(
    const FunctionMapping& function_mapping);

}  // namespace latency_model
}  // namespace mlgo

#endif  // COMPILER_OPT_MEMTRACE_COSTMODEL_EXTRACT_CORPUS_SUBSET_LIB_H_
