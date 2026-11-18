#include "compiler_opt/memtrace_costmodel/compare_binary_cfgs_lib.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "compiler_opt/memtrace_costmodel/elf_metadata_parser.h"
#include "compiler_opt/memtrace_costmodel/status_macros.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

namespace mlgo {
namespace latency_model {
namespace {

// Helper utilities for converting LLVM Errors and Expecteds into
// absl equivalents.
absl::Status LlvmErrorToStatus(llvm::Error error) {
  if (!error) return absl::OkStatus();
  std::string error_string;
  llvm::raw_string_ostream error_string_stream(error_string);
  error_string_stream << error;
  return absl::InternalError(error_string);
}

template <typename T>
absl::StatusOr<T> LlvmExpectedToStatusOr(llvm::Expected<T> expected) {
  if (expected) return std::move(*expected);
  return LlvmErrorToStatus(expected.takeError());
}

}  // namespace

struct CfgEdgeComparator {
  bool operator()(const ControlFlowEdge& lhs,
                  const ControlFlowEdge& rhs) const {
    return std::forward_as_tuple(lhs.from_block, lhs.to_block) <
           std::forward_as_tuple(rhs.from_block, rhs.to_block);
  }
};

struct FunctionID {
  uint64_t function_address;
  uint64_t section_index;
};

struct FunctionIDKeyHash {
  size_t operator()(const FunctionID& function_id) const {
    return absl::HashOf(function_id.function_address,
                        function_id.section_index);
  }
};

struct FunctionIDKeyEqual {
  bool operator()(const FunctionID& lhs, const FunctionID& rhs) const {
    return lhs.function_address == rhs.function_address &&
           lhs.section_index == rhs.section_index;
  }
};

absl::StatusOr<
    absl::flat_hash_map<std::string, const std::vector<ControlFlowEdge>>>
LoadBinaryCfgs(absl::string_view binary_path) {
  // Get the mapping of addresses to function names.
  absl::flat_hash_map<FunctionID, absl::string_view, FunctionIDKeyHash,
                      FunctionIDKeyEqual>
      function_id_to_name;

  ASSIGN_OR_RETURN(
      auto unstripped_binary_processor,
      mlgo::latency_model::UnstrippedBinaryProcessor::Create(binary_path));

  RETURN_IF_ERROR(unstripped_binary_processor->ProcessBBAddrMap(
      [&function_id_to_name](
          const mlgo::latency_model::UnstrippedBinaryProcessor::FunctionBBInfo&
              bb_info) {
        FunctionID function_id = {.function_address = bb_info.function_address,
                                  .section_index = bb_info.section_index};
        function_id_to_name.emplace(function_id, bb_info.function_name);
      }));

  // Load the executable through the LLVM APIs.
  ASSIGN_OR_RETURN(
      llvm::object::OwningBinary<llvm::object::Binary> object_binary,
      LlvmExpectedToStatusOr(llvm::object::createBinary(binary_path)));
  llvm::object::ELFObjectFileBase* elf_object =
      llvm::cast<llvm::object::ELFObjectFileBase>(object_binary.getBinary());

  // Load the CFG into the map.
  absl::flat_hash_map<std::string, const std::vector<ControlFlowEdge>>
      binary_cfgs;

  for (const auto& section : elf_object->sections()) {
    if (!section.isText()) continue;

    std::vector<llvm::object::PGOAnalysisMap> pgo_analysis_maps;
    ASSIGN_OR_RETURN(std::vector<llvm::object::BBAddrMap> bb_addr_maps,
                     LlvmExpectedToStatusOr(elf_object->readBBAddrMap(
                         section.getIndex(), &pgo_analysis_maps)));

    for (const auto& [bb_addr_map, pgo_analysis_map] :
         llvm::zip(bb_addr_maps, pgo_analysis_maps)) {
      FunctionID function_id = {
          .function_address = bb_addr_map.getFunctionAddress(),
          .section_index = section.getIndex()};
      const auto function_name_it = function_id_to_name.find(function_id);
      if (function_name_it == function_id_to_name.end()) {
        // LOG(WARNING)
        //     << "Failed to find a symbol name for function at address 0x"
        //     << absl::Hex(bb_addr_map.getFunctionAddress()) << " in section "
        //     << section.getIndex() << ", skipping.";
        continue;
      }
      absl::string_view function_name = function_name_it->second;

      std::vector<ControlFlowEdge> function_edges;

      for (const auto& [bb_entry, pgo_bb_entry] :
           llvm::zip(bb_addr_map.getBBEntries(), pgo_analysis_map.BBEntries)) {
        for (const auto& bb_successor_entry : pgo_bb_entry.Successors) {
          ControlFlowEdge new_cfg_edge = {.from_block = bb_entry.ID,
                                          .to_block = bb_successor_entry.ID};
          function_edges.push_back(new_cfg_edge);
        }
      }

      absl::c_sort(function_edges, CfgEdgeComparator());

      binary_cfgs.emplace(function_name, std::move(function_edges));
    }
  }

  return binary_cfgs;
}

absl::StatusOr<
    absl::flat_hash_map<std::string, const std::vector<ControlFlowEdge>>>
LoadCorpusCfgs(absl::Span<const std::string> module_paths) {
  absl::flat_hash_map<std::string, const std::vector<ControlFlowEdge>>
      binary_cfgs;

  for (const std::string& module_path : module_paths) {
    ASSIGN_OR_RETURN(auto module_cfgs, LoadBinaryCfgs(module_path));

    binary_cfgs.insert(module_cfgs.begin(), module_cfgs.end());
  }

  return binary_cfgs;
}

bool AreCfgsDifferent(absl::Span<const ControlFlowEdge> function_a_cfg,
                      absl::Span<const ControlFlowEdge> function_b_cfg) {
  if (function_a_cfg.size() != function_b_cfg.size()) {
    return true;
  }

  for (size_t i = 0; i < function_a_cfg.size(); ++i) {
    if (function_a_cfg[i].from_block != function_b_cfg[i].from_block ||
        function_a_cfg[i].to_block != function_b_cfg[i].to_block) {
      LOG(INFO) << "Expected CFG edge (" << function_a_cfg[i].from_block << ","
                << function_a_cfg[i].to_block << ") and ("
                << function_b_cfg[i].from_block << ","
                << function_b_cfg[i].to_block << ") to be the same.";
      return true;
    }
  }

  return false;
}

}  // namespace latency_model
}  // namespace mlgo
