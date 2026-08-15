#include "compiler_opt/memtrace_costmodel/sort_bb_traces_lib.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "compiler_opt/memtrace_costmodel/mbb_trace.proto.h"
#include "riegeli/bytes/file_reader.h"
#include "riegeli/bytes/file_writer.h"
#include "riegeli/records/record_reader.h"
#include "riegeli/records/record_writer.h"

namespace mlgo {
namespace latency_model {

std::vector<TraceInfo> LoadTraceInfo(absl::string_view trace_file_path) {
  riegeli::RecordReader trace_reader(
      riegeli::Maker<riegeli::FileReader>(std::string(trace_file_path)));
  mlgo::latency_model::MbbTrace mbb_trace;
  std::vector<TraceInfo> trace_info;
  size_t index = 0;
  while (trace_reader.ReadRecord(mbb_trace)) {
    trace_info.push_back(
        TraceInfo{.index = index, .size = mbb_trace.mbbs_size()});
    ++index;
  }
  QCHECK(trace_reader.Close()) << trace_reader.status();
  return trace_info;
}

std::vector<size_t> GetTraceIndexToShardMap(std::vector<TraceInfo>& traces_info,
                                            size_t shard_count) {
  int trace_count = traces_info.back().index + 1;
  std::sort(
      traces_info.begin(), traces_info.end(),
      [](const TraceInfo& a, const TraceInfo& b) { return a.size > b.size; });

  std::vector<size_t> index_to_shard_map(trace_count);
  int current_shard_index = 0;
  for (int i = 0; i < shard_count; ++i) {
    std::vector<MbbTrace> current_shard;
    for (int j = i; j < traces_info.size(); j += shard_count) {
      index_to_shard_map[traces_info[j].index] = current_shard_index;
    }
    ++current_shard_index;
  }

  return index_to_shard_map;
}

void WriteOutShards(absl::string_view trace_file_path, int shard_count,
                    const std::vector<size_t>& trace_shards,
                    absl::string_view output_path_template,
                    int compression_level) {
  std::vector<std::unique_ptr<riegeli::RecordWriter<riegeli::FileWriter<>>>>
      shard_writers;
  for (int i = 0; i < shard_count; ++i) {
    shard_writers.push_back(
        std::make_unique<riegeli::RecordWriter<riegeli::FileWriter<>>>(
            riegeli::Maker<riegeli::FileWriter<>>(
                (absl::StrCat(output_path_template, i, ".pb"))),
            riegeli::RecordWriterBase::Options().set_zstd()));
  }

  int index = 0;
  mlgo::latency_model::MbbTrace mbb_trace;
  riegeli::RecordReader trace_reader(
      riegeli::Maker<riegeli::FileReader>(std::string(trace_file_path)));
  while (trace_reader.ReadRecord(mbb_trace)) {
    std::unique_ptr<riegeli::RecordWriter<riegeli::FileWriter<>>>&
        shard_writer = shard_writers[trace_shards[index]];
    shard_writer->WriteRecord(mbb_trace);
    ++index;
  }
  QCHECK(trace_reader.Close()) << trace_reader.status();
  for (int i = 0; i < shard_writers.size(); ++i) {
    QCHECK(shard_writers[i]->Close()) << shard_writers[i]->status();
  }
}

}  // namespace latency_model
}  // namespace mlgo
