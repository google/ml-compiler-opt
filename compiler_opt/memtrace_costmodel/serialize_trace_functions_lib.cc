#include "compiler_opt/memtrace_costmodel/serialize_trace_functions_lib.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "compiler_opt/memtrace_costmodel/basic_block_trace.h"
#include "compiler_opt/memtrace_costmodel/mbb_trace.proto.h"
#include "compiler_opt/memtrace_costmodel/serialized_mbbs.proto.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlgo {
namespace latency_model {

absl::flat_hash_set<MachineBbId, MachineBBIDKeyHash, MachineBBIDKeyEqual>
GetRequiredMbbs(absl::Span<const MbbTrace> mbb_traces,
                absl::Span<const std::string> functions_to_exclude,
                const FunctionMapping& function_names_to_id) {
  absl::flat_hash_set<MachineBbId, MachineBBIDKeyHash, MachineBBIDKeyEqual>
      unique_bbs;
  absl::flat_hash_set<uint32_t> functions_ids_to_exclude;

  for (const std::string& function_to_exclude : functions_to_exclude) {
    const auto function_id_it =
        function_names_to_id.function_ids().find(function_to_exclude);
    if (function_id_it == function_names_to_id.function_ids().end()) {
      LOG(WARNING) << "Failed to find an ID for function "
                   << function_to_exclude << " in the function mapping";
      continue;
    }

    functions_ids_to_exclude.emplace(function_id_it->second);
  }

  for (const MbbTrace& mbb_trace : mbb_traces) {
    for (const MachineBbId& basic_block : mbb_trace.mbbs()) {
      if (functions_ids_to_exclude.contains(basic_block.function_id())) {
        continue;
      }

      // Blocks without function IDs are shared object traces. We want to skip
      // them because they are held within the trace.
      if (!basic_block.has_function_id()) {
        continue;
      }

      unique_bbs.emplace(basic_block);
    }
  }

  return unique_bbs;
}

SerializedMbbs GetSerializedMbbs(
    const std::vector<std::string>& bitcode_paths,
    const std::string& target_triple,
    const FunctionMapping& function_names_to_id,
    const absl::flat_hash_set<MachineBbId, MachineBBIDKeyHash,
                              MachineBBIDKeyEqual>& mbbs_to_include) {
  SerializedMbbs mbbs_to_return;

  CorpusApplicationToBbDisassembler corpus_to_bb_contents(
      target_triple, bitcode_paths, /*store_block_contents=*/true);
  corpus_to_bb_contents.LoadBasicBlocks(function_names_to_id);

  for (const MachineBbId& entry_id : mbbs_to_include) {
    llvm::ArrayRef<uint8_t> entry_contents =
        corpus_to_bb_contents.GetEntryContents(entry_id);

    *mbbs_to_return.add_mbb_ids() = entry_id;

    absl::string_view current_instruction_data(
        reinterpret_cast<const char*>(entry_contents.data()),
        entry_contents.size());
    mbbs_to_return.add_mbb_bytes(current_instruction_data);
  }

  return mbbs_to_return;
}

std::vector<std::string> LoadExcludedFunctions(
    absl::string_view excluded_functions_list) {
  std::vector<std::string> excluded_functions;
  std::ifstream file((std::string(excluded_functions_list)));
  QCHECK(file) << "Failed to open excluded functions list: "
               << excluded_functions_list;
  std::string line;
  while (std::getline(file, line)) {
    // Remove carriage return if present (similar to REMOVE_LINEFEED)
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    excluded_functions.push_back(line);
  }
  return excluded_functions;
}

}  // namespace latency_model
}  // namespace mlgo
