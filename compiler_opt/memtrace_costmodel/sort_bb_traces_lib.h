#ifndef COMPILER_OPT_MEMTRACE_COSTMODEL_SORT_BB_TRACES_LIB_H_
#define COMPILER_OPT_MEMTRACE_COSTMODEL_SORT_BB_TRACES_LIB_H_

#include <cstddef>
#include <vector>

#include "absl/strings/string_view.h"
#include "compiler_opt/memtrace_costmodel/mbb_trace.proto.h"

namespace mlgo {
namespace latency_model {

struct TraceInfo {
  size_t index;
  int size;
};

std::vector<TraceInfo> LoadTraceInfo(absl::string_view trace_file_path);

// The traces are expected to be passed in the order in which they are parsed
// (i.e., with the highest index trace at the back).
std::vector<size_t> GetTraceIndexToShardMap(std::vector<TraceInfo>& traces_info,
                                            size_t shard_count);

void WriteOutShards(absl::string_view trace_file_path, int shard_count,
                    const std::vector<size_t>& trace_shards,
                    absl::string_view output_path_template,
                    int compression_level = 1 << 16);

}  // namespace latency_model
}  // namespace mlgo

#endif  // COMPILER_OPT_MEMTRACE_COSTMODEL_SORT_BB_TRACES_LIB_H_
