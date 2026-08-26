#include <cstddef>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "compiler_opt/memtrace_costmodel/mbb_trace.proto.h"
#include "compiler_opt/memtrace_costmodel/sort_bb_traces_lib.h"

ABSL_FLAG(std::string, input_trace, "", "The trace to take as input.");
ABSL_FLAG(std::string, output_path, "", "The output path pattern.");
ABSL_FLAG(int, shard_count, 1, "The shard count.");
ABSL_FLAG(
    int, compression_level, 1 << 16,
    "The compression level to use while writing the basic block trace protos.");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  if (absl::GetFlag(FLAGS_input_trace).empty()) {
    LOG(QFATAL) << "--input_trace must be set.";
  }
  if (absl::GetFlag(FLAGS_output_path).empty()) {
    LOG(QFATAL) << "--output_path must be set.";
  }

  const std::string input_trace = absl::GetFlag(FLAGS_input_trace);
  const std::string output_path = absl::GetFlag(FLAGS_output_path);
  const int shard_count = absl::GetFlag(FLAGS_shard_count);
  const int compression_level = absl::GetFlag(FLAGS_compression_level);

  std::vector<mlgo::latency_model::TraceInfo> trace_info =
      mlgo::latency_model::LoadTraceInfo(input_trace);
  LOG(INFO) << "Finished loading trace info.";

  std::vector<size_t> trace_index_to_shard_map =
      mlgo::latency_model::GetTraceIndexToShardMap(trace_info, shard_count);
  LOG(INFO) << "Finished planning shards.";

  mlgo::latency_model::WriteOutShards(input_trace, shard_count,
                                      trace_index_to_shard_map, output_path,
                                      compression_level);

  return 0;
}
