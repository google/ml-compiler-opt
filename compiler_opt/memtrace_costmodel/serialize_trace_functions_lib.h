#ifndef COMPILER_OPT_MEMTRACE_COSTMODEL_SERIALIZE_TRACE_FUNCTIONS_LIB_H_
#define COMPILER_OPT_MEMTRACE_COSTMODEL_SERIALIZE_TRACE_FUNCTIONS_LIB_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "compiler_opt/memtrace_costmodel/basic_block_trace.h"
#include "compiler_opt/memtrace_costmodel/mbb_trace.proto.h"
#include "compiler_opt/memtrace_costmodel/serialized_mbbs.proto.h"

namespace mlgo {
namespace latency_model {

absl::flat_hash_set<MachineBbId, MachineBBIDKeyHash, MachineBBIDKeyEqual>
GetRequiredMbbs(absl::Span<const MbbTrace> mbb_traces,
                absl::Span<const std::string> functions_to_exclude,
                const FunctionMapping& function_names_to_id);

SerializedMbbs GetSerializedMbbs(
    const std::vector<std::string>& bitcode_paths,
    const std::string& target_triple,
    const FunctionMapping& function_names_to_id,
    const absl::flat_hash_set<MachineBbId, MachineBBIDKeyHash,
                              MachineBBIDKeyEqual>& mbbs_to_include);

std::vector<std::string> LoadExcludedFunctions(
    absl::string_view excluded_functions_list);

}  // namespace latency_model
}  // namespace mlgo

#endif  // COMPILER_OPT_MEMTRACE_COSTMODEL_SERIALIZE_TRACE_FUNCTIONS_LIB_H_
