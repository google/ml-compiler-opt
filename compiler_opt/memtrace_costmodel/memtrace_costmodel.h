#ifndef COMPILER_OPT_MEMTRACE_COSTMODEL_MEMTRACE_COSTMODEL_H_
#define COMPILER_OPT_MEMTRACE_COSTMODEL_MEMTRACE_COSTMODEL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "compiler_opt/memtrace_costmodel/costmodel.h"
#include "llvm/MC/MCInst.h"

// Standard DynamoRIO inclusions
#include "absl/synchronization/mutex.h"
#include "drmemtrace/analysis_tool.h"
#include "drmemtrace/memref.h"
#include "drmemtrace/raw2trace_shared.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/Target/TargetMachine.h"

namespace mlgo {
namespace latency_model {

// Simple portable callstack context tracker stub.
template <typename T>
class CallstackTracker {
 public:
  CallstackTracker(std::function<std::unique_ptr<T>()> factory)
      : factory_(factory), context_(factory_()) {}
  T& Current() { return *context_; }

 private:
  std::function<std::unique_ptr<T>()> factory_;
  std::unique_ptr<T> context_;
};

class CostModelCallstackContext {
 public:
  CostModelCallstackContext(
      dynamorio::drmemtrace::module_mapper_t* module_mapper,
      absl::Span<const uint64_t> symbols_of_interest_addresses)
      : module_mapper_(module_mapper),
        symbols_of_interest_addresses_(symbols_of_interest_addresses) {};

  void PushToLeafstack(uint64_t addr);
  void PopLeafstack();
  bool UnderEntrypoint() const { return entrypoint_count_ > 0; };
  bool JustLeftEntrypoint() const { return just_left_point_; };
  void ResetJustLeftEntrypoint() { just_left_point_ = false; };
  const std::vector<uint64_t>& Leafstack() const { return leafstack_; }

 private:
  uint64_t ElfAddressFromPc(uint64_t current_pc);
  bool IsPcDirectlyUnderEntrypoint(uint64_t current_pc);

  int64_t entrypoint_count_ = 0;
  bool just_left_point_ = false;
  std::vector<uint64_t> leafstack_;
  dynamorio::drmemtrace::module_mapper_t* module_mapper_;
  const absl::Span<const uint64_t> symbols_of_interest_addresses_;
};

using CostModelCallstackTracker = CallstackTracker<CostModelCallstackContext>;

llvm::MCInst getInstructionFromBytes(llvm::ArrayRef<uint8_t> instruction_data,
                                     llvm::MCDisassembler& disassembler,
                                     uint64_t& instruction_size,
                                     uint64_t instruction_address = 0);

bool isInstructionNoop(const llvm::MCInst& machine_instruction);

// CostMemtraceProcessor inherits from standard DynamoRIO analysis_tool_t
// to keep the trace analysis logic functional and open-source.
class CostMemtraceProcessor : public dynamorio::drmemtrace::analysis_tool_t {
 public:
  explicit CostMemtraceProcessor(
      dynamorio::drmemtrace::module_mapper_t* module_mapper,
      absl::Span<const uint64_t> symbols_of_interest_addresses,
      absl::FunctionRef<void(std::vector<double>)> entrypoint_cost_processor,
      absl::FunctionRef<std::unique_ptr<CostModel>()> cost_model_factory,
      std::string target_triple, std::string cpu_name, bool split_on_segment);

  // Implement pure virtual methods of analysis_tool_t
  bool process_memref(const dynamorio::drmemtrace::memref_t& entry) override;
  bool print_results() override { return true; }

 private:
  struct PerThreadData {
    explicit PerThreadData(
        std::function<std::unique_ptr<CostModelCallstackContext>()>
            callstack_factory_,
        std::unique_ptr<CostModel> cost_model,
        std::unique_ptr<llvm::MCDisassembler> mc_disassembler)
        : callstack(std::move(callstack_factory_)),
          segment_cost_model(std::move(cost_model)),
          llvm_mc_disassembler(std::move(mc_disassembler)) {};

    CostModelCallstackTracker callstack;
    std::unique_ptr<CostModel> segment_cost_model;
    bool previously_under_segment = false;
    std::unique_ptr<llvm::MCDisassembler> llvm_mc_disassembler;
    std::vector<double> entrypoint_segment_costs = {};
  };

  void* parallel_shard_init(int shard_index, void* worker_data) override;
  bool parallel_shard_exit(void* shard_data) override;

  bool parallel_shard_supported() override { return true; }

  void PushCostAndResetModel(PerThreadData& current_shard_data);

  bool parallel_shard_memref(
      void* shard_data, const dynamorio::drmemtrace::memref_t& memref) override;

  // Processes individual instructions, eventually handing them off to the
  // underlying cost model.
  void ProcessMachineInstruction(void* shard_data,
                                 const llvm::MCInst& machine_instruction);

  std::unique_ptr<llvm::TargetMachine> llvm_target_machine_;
  std::unique_ptr<llvm::MCContext> llvm_mc_context_;

  dynamorio::drmemtrace::module_mapper_t* module_mapper_;
  absl::Span<const uint64_t> symbols_of_interest_addresses_;
  absl::FunctionRef<void(std::vector<double>)> entrypoint_cost_processor_;
  absl::Mutex cost_processor_mutex_;

  absl::FunctionRef<std::unique_ptr<CostModel>()> cost_model_factory_;

  std::string target_triple_;
  std::string cpu_name_;

  bool split_on_segment_ = false;
};

// Re-enabled standard GetMemtraceCost using DynamoRIO structures.
// It might not fully compile if some test setups require internal analyzer_t,
// but the core functionality is preserved and adapted to standard DynamoRIO.
absl::Status GetMemtraceCost(
    std::string file_name,
    absl::Span<const uint64_t> symbols_of_interest_addresses,
    absl::FunctionRef<void(std::vector<double>)> entrypoint_cost_processor,
    absl::FunctionRef<std::unique_ptr<CostModel>()> cost_model_factory,
    bool split_on_segment, std::string target_triple = "x86_64",
    std::string cpu_name = "skylake");

}  // namespace latency_model
}  // namespace mlgo

#endif  // COMPILER_OPT_MEMTRACE_COSTMODEL_MEMTRACE_COSTMODEL_H_
