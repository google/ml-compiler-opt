#ifndef COMPILER_OPT_MEMTRACE_COSTMODEL_BASIC_BLOCK_TRACE_H_
#define COMPILER_OPT_MEMTRACE_COSTMODEL_BASIC_BLOCK_TRACE_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "compiler_opt/memtrace_costmodel/mbb_trace.proto.h"
#include "compiler_opt/memtrace_costmodel/serialized_mbbs.proto.h"
#include "drmemtrace/analysis_tool.h"
#include "drmemtrace/memref.h"
#include "drmemtrace/raw2trace_shared.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Target/TargetMachine.h"

namespace mlgo {
namespace latency_model {

// We include has_function_id() in the hash and equality checks as we use
// MachineBbIds without function IDs to represent direct traces so they do
// not overlap with a potential function that has ID zero.

struct MachineBBIDKeyHash {
  size_t operator()(const MachineBbId& machine_bbid) const {
    return absl::HashOf(machine_bbid.has_function_id(),
                        machine_bbid.function_id(),
                        machine_bbid.basic_block_id(), machine_bbid.entry_id());
  }
};

struct MachineBBIDKeyEqual {
  bool operator()(const MachineBbId& lhs, const MachineBbId& rhs) const {
    return lhs.has_function_id() == rhs.has_function_id() &&
           lhs.function_id() == rhs.function_id() &&
           lhs.basic_block_id() == rhs.basic_block_id() &&
           lhs.entry_id() == rhs.entry_id();
  }
};

struct SectionInfo {
  uint64_t address = 0;
  uint64_t size = 0;
};

struct InstructionInfo {
  llvm::MCInst instruction;
  uint64_t address = 0;
  uint64_t size = 0;
};

class BBMemtraceProcessor : public dynamorio::drmemtrace::analysis_tool_t {
 public:
  // The backing array for symbols_of_interest_addresses needs to outlive the
  // created instance of this class.
  explicit BBMemtraceProcessor(
      const absl::flat_hash_map<uint64_t, MachineBbId>&
          bb_addresses_to_ids_and_functions,
      const std::vector<std::string>& function_id_to_name,
      const dynamorio::drmemtrace::module_mapper_t* module_mapper,
      absl::Span<const uint64_t> symbols_of_interest_addresses,
      absl::FunctionRef<void(const MbbTrace&)>& bb_trace_processor,
      FunctionMapping& function_name_to_id, bool split_on_segment,
      int64_t max_blocks_per_segment,
      std::vector<SectionInfo>&& sections_with_bb_info,
      std::string expected_build_id, std::string binary_path)
      : bb_addresses_to_ids_and_functions_(bb_addresses_to_ids_and_functions),
        function_id_to_name_(function_id_to_name),
        module_mapper_(module_mapper),
        symbols_of_interest_addresses_(symbols_of_interest_addresses),
        bb_trace_processor_(bb_trace_processor),
        function_name_to_id_(function_name_to_id),
        split_on_segment_(split_on_segment),
        max_blocks_per_segment_(max_blocks_per_segment),
        sections_with_bb_info_(std::move(sections_with_bb_info)),
        expected_build_id_(std::move(expected_build_id)),
        binary_path_(std::move(binary_path)) {};

 private:
  struct PerShardData {
    PerShardData() = default;

    absl::Mutex mutex;
    MbbTrace current_trace_data ABSL_GUARDED_BY(mutex);
    std::vector<MbbTrace> entrypoint_segments ABSL_GUARDED_BY(mutex);
    bool previously_under_segment ABSL_GUARDED_BY(mutex) = false;
    bool inside_shared_object ABSL_GUARDED_BY(mutex) = false;
    absl::flat_hash_map<std::string, uint32_t> function_name_to_id
        ABSL_GUARDED_BY(mutex);
  };

  void* parallel_shard_init(int shard_index, void* worker_data) override;
  bool parallel_shard_exit(void* shard_data) override;

  bool process_memref(const dynamorio::drmemtrace::memref_t& memref) override;

  bool parallel_shard_supported() override { return true; }

  bool parallel_shard_memref(
      void* shard_data, const dynamorio::drmemtrace::memref_t& memref) override;

  bool print_results() override { return true; }

  const absl::flat_hash_map<uint64_t, MachineBbId>&
      bb_addresses_to_ids_and_functions_;
  const std::vector<std::string>& function_id_to_name_;
  const dynamorio::drmemtrace::module_mapper_t* module_mapper_;
  const absl::Span<const uint64_t> symbols_of_interest_addresses_;
  absl::FunctionRef<void(const MbbTrace&)>& bb_trace_processor_;
  absl::Mutex bb_trace_processor_mutex_;
  absl::Mutex function_name_to_id_mutex_;
  FunctionMapping& function_name_to_id_
      ABSL_GUARDED_BY(function_name_to_id_mutex_);
  bool split_on_segment_ = false;
  const int64_t max_blocks_per_segment_ = 0;
  std::vector<SectionInfo> sections_with_bb_info_;
  std::string expected_build_id_;
  std::string binary_path_;

  uint32_t current_trace_id_ ABSL_GUARDED_BY(current_trace_id_mutex_) = 0;
  absl::Mutex current_trace_id_mutex_;
};

absl::Status GetBasicBlockTracesFromDirectory(
    absl::string_view trace_dir, absl::string_view binary_path,
    absl::FunctionRef<void(const MbbTrace&)> bb_trace_processor,
    absl::Span<const uint64_t> symbols_of_interest_addresses,
    FunctionMapping& function_name_to_id, bool split_on_segment,
    int64_t max_blocks_per_segment = std::numeric_limits<int64_t>::max());

absl::Status GetBasicBlockTracesFromDirectory(
    absl::string_view trace_dir, absl::string_view binary_path,
    absl::FunctionRef<void(const MbbTrace&)> bb_trace_processor,
    absl::Span<const std::string> symbols_of_interest,
    FunctionMapping& function_name_to_id, bool split_on_segment,
    int64_t max_blocks_per_segment = std::numeric_limits<int64_t>::max());

absl::Status WriteBasicBlockTraces(
    absl::string_view trace_dir, absl::string_view binary_path,
    absl::string_view output_folder,
    absl::Span<const std::string> symbols_of_interest, bool split_on_segment,
    int64_t max_blocks_per_segment = std::numeric_limits<int64_t>::max());

absl::StatusOr<absl::flat_hash_map<uint64_t, MachineBbId>>
GetBbAddressesToIdsMap(absl::string_view binary_path,
                       std::vector<std::string>& function_id_to_name);

// Takes a application in an implementation-defined form (like a binary or
// corpus) and disassembles individual basic blocks within the application on
// demand with caching.
class ApplicationToBbDisassembler {
 public:
  llvm::ArrayRef<InstructionInfo> GetDisassembledInstructions(
      MachineBbId function_basic_block_id);

  // Calls entry_processor with each block entry found in the basic block
  // at block_address with size block_size. An entry is defined as an
  // instruction where a block can be entered, such as when returning
  // from a call within that block.
  void ProcessAllEntriesInBlock(
      uint64_t block_address, uint32_t block_size,
      absl::FunctionRef<void(uint64_t)> entry_processor) const;

  virtual void LoadBasicBlocks(const FunctionMapping& function_name_to_id);

  void LoadSharedObjectTraces(const MbbTrace& mbb_trace);

  // Loads serialized machine basic blocks into internal state such that they
  // can then be queried using GetDisassembledInstructions.
  void LoadSerializedBbs(const SerializedMbbs& serialized_mbbs);

  virtual ~ApplicationToBbDisassembler();

 protected:
  explicit ApplicationToBbDisassembler(const std::string& target_triple);
  std::unique_ptr<llvm::MCDisassembler> llvm_mc_disassembler_;
  std::unique_ptr<llvm::MCInstrInfo> llvm_mc_instr_info_;

  void ProcessAllEntriesFromBlockContents(
      uint64_t block_start_address,
      absl::FunctionRef<void(uint64_t, std::vector<InstructionInfo>&&,
                             llvm::ArrayRef<uint8_t>)>
          entry_processor,
      llvm::ArrayRef<uint8_t> block_contents,
      uint64_t expected_block_size) const;

  absl::flat_hash_map<MachineBbId, std::vector<InstructionInfo>,
                      MachineBBIDKeyHash, MachineBBIDKeyEqual>
      disassembled_instructions_;

  absl::Mutex disassembling_instructions_mutex_;

 private:
  void PopulateLlvmHelpers(const std::string& target_triple);

  std::unique_ptr<llvm::TargetMachine> llvm_target_machine_;
  std::unique_ptr<llvm::MCContext> llvm_mc_context_;
};

class BinaryApplicationToBbDisassembler : public ApplicationToBbDisassembler {
 public:
  explicit BinaryApplicationToBbDisassembler(const std::string& target_triple,
                                             const std::string& binary_path);

  void ProcessAllEntriesInBlock(
      uint64_t block_address, uint32_t block_size,
      absl::FunctionRef<void(uint64_t, std::vector<InstructionInfo>,
                             llvm::ArrayRef<uint8_t>)>
          entry_processor) const;

  void LoadBasicBlocks(const FunctionMapping& function_name_to_id) override;

 private:
  // Gets the contents of a basic block identified by its address and
  // size. This is intended for processing through all the basic blocks
  // in a binary so that maps can be set up appropriately for partial
  // basic blocks.
  llvm::ArrayRef<uint8_t> GetBlockContentsFromAddress(
      uint64_t block_address, uint32_t block_size) const;

  llvm::object::OwningBinary<llvm::object::Binary> binary_;

  absl::btree_map<uint64_t, llvm::object::SectionRef> address_to_section_;

  std::string binary_path_;
};

// Takes an application in the form of a corpus and implements functions
// to find where specific blocks are within the corpus and to grab the bytes
// for those blocks so they can be disassembled and used for cost modelling.
class CorpusApplicationToBbDisassembler : public ApplicationToBbDisassembler {
 public:
  explicit CorpusApplicationToBbDisassembler(
      const std::string& target_triple,
      const std::vector<std::string>& module_paths,
      bool store_block_contents = false);

  void ProcessAllEntriesInBlock(
      uint64_t bb_offset, uint32_t bb_size, llvm::object::SectionRef bb_section,
      absl::FunctionRef<void(uint64_t, std::vector<InstructionInfo>,
                             llvm::ArrayRef<uint8_t>)>
          entry_processor) const;

  void LoadBasicBlocks(const FunctionMapping& function_name_to_id) override;

  llvm::ArrayRef<uint8_t> GetEntryContents(MachineBbId function_basic_block_id);

 private:
  llvm::ArrayRef<uint8_t> GetBlockContentsFromSectionOffset(
      uint64_t bb_offset, uint32_t bb_size,
      llvm::object::SectionRef bb_section) const;

  const std::vector<std::string>& module_paths_;
  const bool store_block_contents_ = false;

  absl::flat_hash_map<MachineBbId, std::vector<uint8_t>, MachineBBIDKeyHash,
                      MachineBBIDKeyEqual>
      entry_contents_;
};

}  // namespace latency_model
}  // namespace mlgo

#endif  // COMPILER_OPT_MEMTRACE_COSTMODEL_BASIC_BLOCK_TRACE_H_
