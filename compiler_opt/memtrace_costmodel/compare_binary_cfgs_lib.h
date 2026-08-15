#ifndef COMPILER_OPT_MEMTRACE_COSTMODEL_COMPARE_BINARY_CFGS_LIB_H_
#define COMPILER_OPT_MEMTRACE_COSTMODEL_COMPARE_BINARY_CFGS_LIB_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace mlgo {
namespace latency_model {

struct ControlFlowEdge {
  uint32_t from_block = 0;
  uint32_t to_block = 0;
};

absl::StatusOr<
    absl::flat_hash_map<std::string, const std::vector<ControlFlowEdge>>>
LoadBinaryCfgs(absl::string_view binary_path);

absl::StatusOr<
    absl::flat_hash_map<std::string, const std::vector<ControlFlowEdge>>>
LoadCorpusCfgs(absl::Span<const std::string> module_paths);

// Detects if the CFGs are different by directly comparing the CFG Edge
// vectors. They are assumed to be sorted.
bool AreCfgsDifferent(absl::Span<const ControlFlowEdge> function_a_cfg,
                      absl::Span<const ControlFlowEdge> function_b_cfg);

}  // namespace latency_model
}  // namespace mlgo

#endif  // COMPILER_OPT_MEMTRACE_COSTMODEL_COMPARE_BINARY_CFGS_LIB_H_
