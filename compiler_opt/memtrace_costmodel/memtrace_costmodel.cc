#include "compiler_opt/memtrace_costmodel/memtrace_costmodel.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "compiler_opt/memtrace_costmodel/costmodel.h"

// Standard DynamoRIO inclusions
#include "drmemtrace/analysis_tool.h"
#include "drmemtrace/analyzer.h"
#include "drmemtrace/memref.h"
#include "drmemtrace/raw2trace_shared.h"

// Undefine conflicting DynamoRIO macro before including LLVM headers
#undef X86
#undef X86_64

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/lib/Target/X86/MCTargetDesc/X86MCTargetDesc.h"

namespace mlgo {
namespace latency_model {

using ::dynamorio::drmemtrace::memref_t;

absl::Status GetMemtraceCost(
    std::string file_name,
    absl::Span<const uint64_t> symbols_of_interest_addresses,
    absl::FunctionRef<void(std::vector<double>)> entrypoint_cost_processor,
    absl::FunctionRef<std::unique_ptr<CostModel>()> cost_model_factory,
    bool split_on_segment, std::string target_triple, std::string cpu_name) {
  for (uint64_t symbol_of_interest_address : symbols_of_interest_addresses) {
    CHECK_NE(symbol_of_interest_address, 0);
  }

  // The module mapper is used to read the module list and map instruction
  // addresses. For offline traces, it can be created from the trace directory
  // metadata.
  std::unique_ptr<dynamorio::drmemtrace::module_mapper_t> module_mapper =
      dynamorio::drmemtrace::module_mapper_t::create(nullptr);

  auto memtrace_processor = std::make_unique<CostMemtraceProcessor>(
      module_mapper.get(), symbols_of_interest_addresses,
      entrypoint_cost_processor, cost_model_factory, target_triple, cpu_name,
      split_on_segment);

  std::vector<dynamorio::drmemtrace::analysis_tool_t*> tools;
  tools.push_back(memtrace_processor.get());

  // Reimplemented using standard, portable DynamoRIO analyzer_t.
  dynamorio::drmemtrace::analyzer_t analyzer(file_name, &tools[0], 1);
  if (!analyzer.run()) {
    return absl::InternalError("Failed to run memtrace analyzer");
  }

  return absl::OkStatus();
}

void CostModelCallstackContext::PushToLeafstack(uint64_t addr) {
  if (IsPcDirectlyUnderEntrypoint(addr)) {
    ++entrypoint_count_;
  }
  leafstack_.push_back(addr);
}

void CostModelCallstackContext::PopLeafstack() {
  if (leafstack_.empty()) return;
  if (IsPcDirectlyUnderEntrypoint(leafstack_.back())) {
    --entrypoint_count_;
    if (entrypoint_count_ == 0) {
      just_left_point_ = true;
    }
  }
  leafstack_.pop_back();
}

uint64_t CostModelCallstackContext::ElfAddressFromPc(uint64_t current_pc) {
  // Standard DynamoRIO traces are post-processed by raw2trace, which maps all
  // runtime PCs back to binary segment offsets (ELF PCs), resolving ASLR.
  // Thus, trace PCs are already identical to ELF addresses.
  // If further mapping is needed, the module_mapper_t can be used to map trace
  // addresses.
  if (module_mapper_ != nullptr) {
    app_pc mapped_pc = module_mapper_->find_mapped_trace_address(
        reinterpret_cast<app_pc>(current_pc));
    if (mapped_pc != nullptr) {
      return reinterpret_cast<uint64_t>(mapped_pc);
    }
  }
  return current_pc;
}

bool CostModelCallstackContext::IsPcDirectlyUnderEntrypoint(
    uint64_t current_pc) {
  uint64_t elf_address = ElfAddressFromPc(current_pc);
  for (uint64_t symbol_of_interest_address : symbols_of_interest_addresses_) {
    if (elf_address == symbol_of_interest_address) {
      return true;
    }
  }
  return false;
}

llvm::MCInst getInstructionFromBytes(llvm::ArrayRef<uint8_t> instruction_data,
                                     llvm::MCDisassembler& disassembler,
                                     uint64_t& instruction_size,
                                     uint64_t instruction_address) {
  llvm::MCInst instruction;
  std::string disassembler_output_buffer;
  llvm::raw_string_ostream output_stream(disassembler_output_buffer);

  const llvm::MCDisassembler::DecodeStatus status = disassembler.getInstruction(
      instruction, instruction_size, instruction_data, instruction_address,
      output_stream);
  output_stream.flush();
  QCHECK_EQ(status, llvm::MCDisassembler::DecodeStatus::Success)
      << disassembler_output_buffer;

  return instruction;
}

CostMemtraceProcessor::CostMemtraceProcessor(
    dynamorio::drmemtrace::module_mapper_t* module_mapper,
    absl::Span<const uint64_t> symbols_of_interest_addresses,
    absl::FunctionRef<void(std::vector<double>)> entrypoint_cost_processor,
    absl::FunctionRef<std::unique_ptr<CostModel>()> cost_model_factory,
    std::string target_triple, std::string cpu_name, bool split_on_segment)
    : module_mapper_(module_mapper),
      symbols_of_interest_addresses_(symbols_of_interest_addresses),
      entrypoint_cost_processor_(entrypoint_cost_processor),
      cost_model_factory_(cost_model_factory),
      target_triple_(target_triple),
      cpu_name_(cpu_name),
      split_on_segment_(split_on_segment) {
  // Initialize LLVM.
  std::string possible_lookup_error;
  llvm::Triple triple(target_triple_);
  const llvm::Target* const llvm_target =
      llvm::TargetRegistry::lookupTarget(triple, possible_lookup_error);
  QCHECK(llvm_target) << possible_lookup_error;

  llvm::TargetOptions llvm_target_options;

  llvm_target_machine_.reset(llvm_target->createTargetMachine(
      triple, /*CPU*/ "", /*Features*/ "", llvm_target_options, std::nullopt));
  QCHECK(llvm_target_machine_);
  llvm_mc_context_ = std::make_unique<llvm::MCContext>(
      llvm_target_machine_->getTargetTriple(),
      llvm_target_machine_->getMCAsmInfo(),
      llvm_target_machine_->getMCRegisterInfo(),
      llvm_target_machine_->getMCSubtargetInfo());
  QCHECK(llvm_mc_context_);
}

void* CostMemtraceProcessor::parallel_shard_init(int shard_index,
                                                 void* worker_data) {
  std::string possible_lookup_error;
  const llvm::Target* const llvm_target = llvm::TargetRegistry::lookupTarget(
      llvm::Triple(target_triple_), possible_lookup_error);
  QCHECK(llvm_target);

  std::unique_ptr<CostModel> cost_model = cost_model_factory_();
  PerThreadData* thread_data = new PerThreadData(
      [this]() {
        return std::make_unique<CostModelCallstackContext>(
            module_mapper_, symbols_of_interest_addresses_);
      },
      std::move(cost_model),
      std::unique_ptr<llvm::MCDisassembler>(llvm_target->createMCDisassembler(
          llvm_target_machine_->getMCSubtargetInfo(), *llvm_mc_context_)));
  return reinterpret_cast<void*>(thread_data);
}

bool CostMemtraceProcessor::parallel_shard_exit(void* shard_data) {
  PerThreadData* current_shard_data =
      reinterpret_cast<PerThreadData*>(shard_data);
  delete current_shard_data;
  return true;
}

bool CostMemtraceProcessor::process_memref(const memref_t& memref) {
  LOG(QFATAL) << "Intentionally not implemented";
  return false;
}

void CostMemtraceProcessor::PushCostAndResetModel(
    PerThreadData& current_shard_data) {
  current_shard_data.entrypoint_segment_costs.push_back(
      current_shard_data.segment_cost_model->GetCost());
  current_shard_data.segment_cost_model = cost_model_factory_();
}

bool CostMemtraceProcessor::parallel_shard_memref(void* shard_data,
                                                  const memref_t& memref) {
  PerThreadData* current_shard_data =
      reinterpret_cast<PerThreadData*>(shard_data);

  // Context updating logic is performed manually or using a portable tracker.
  // We update the callstack tracker depending on the instruction memref type.
  if (dynamorio::drmemtrace::type_is_instr(memref.instr.type)) {
    // Simple tracking of calls and returns:
    if (memref.instr.type ==
            dynamorio::drmemtrace::TRACE_TYPE_INSTR_DIRECT_CALL ||
        memref.instr.type ==
            dynamorio::drmemtrace::TRACE_TYPE_INSTR_INDIRECT_CALL) {
      current_shard_data->callstack.Current().PushToLeafstack(
          memref.instr.addr);
    } else if (memref.instr.type ==
               dynamorio::drmemtrace::TRACE_TYPE_INSTR_RETURN) {
      current_shard_data->callstack.Current().PopLeafstack();
    }
  }

  // Skip all non-instruction memrefs.
  if (!dynamorio::drmemtrace::type_is_instr(memref.instr.type)) return true;

  if (current_shard_data->previously_under_segment &&
      !current_shard_data->callstack.Current().UnderEntrypoint() &&
      split_on_segment_) {
    // We have just left an entrypoint segment. We need to add the current
    // cost and recreate the cost model, but not return the vector to the user
    // yet.
    PushCostAndResetModel(*current_shard_data);
  }

  if (current_shard_data->callstack.Current().JustLeftEntrypoint()) {
    current_shard_data->callstack.Current().ResetJustLeftEntrypoint();

    // If we are not splitting by segment, then we need to capture the
    // cost for the entire entrypoint here.
    if (!split_on_segment_) {
      PushCostAndResetModel(*current_shard_data);
    }

    {
      absl::MutexLock lock(&cost_processor_mutex_);
      entrypoint_cost_processor_(
          std::move(current_shard_data->entrypoint_segment_costs));
    }
  }

  current_shard_data->previously_under_segment =
      current_shard_data->callstack.Current().UnderEntrypoint();

  // Skip instructions that are not under the entrypoint of interest.
  if (!current_shard_data->callstack.Current().UnderEntrypoint()) return true;

  // Process the instruction into LLVM MCInsts so that we can utilize LLVM
  // tooling for downstream processing. We loop through until the total
  // instruction size is equal to the size of the encoding captured in the
  // memref as we might have multiple MCInsts in a single instruction, for
  // example with a lock prefix before a cmpxchg instruction.
  uint64_t total_instruction_size = 0;
  while (total_instruction_size < memref.instr.size) {
    llvm::ArrayRef<uint8_t> instruction_data(
        reinterpret_cast<const uint8_t*>(memref.instr.encoding +
                                         total_instruction_size),
        memref.instr.size);
    uint64_t instruction_size = 0;
    llvm::MCInst current_instruction = getInstructionFromBytes(
        instruction_data, *current_shard_data->llvm_mc_disassembler,
        instruction_size, memref.instr.addr);

    ProcessMachineInstruction(shard_data, current_instruction);

    total_instruction_size += instruction_size;
  }

  return true;
}

bool isInstructionNoop(const llvm::MCInst& machine_instruction) {
  if (machine_instruction.getOpcode() == llvm::X86::NOOP ||
      machine_instruction.getOpcode() == llvm::X86::NOOPL ||
      machine_instruction.getOpcode() == llvm::X86::NOOPLr ||
      machine_instruction.getOpcode() == llvm::X86::NOOPQ ||
      machine_instruction.getOpcode() == llvm::X86::NOOPQr ||
      machine_instruction.getOpcode() == llvm::X86::NOOPW ||
      machine_instruction.getOpcode() == llvm::X86::NOOPWr) {
    return true;
  }

  return false;
}

void CostMemtraceProcessor::ProcessMachineInstruction(
    void* shard_data, const llvm::MCInst& machine_instruction) {
  // Do not process no-op instructions as we do not currently (at least for
  // MCA) model the (pre)decoder, which means they do not impact the model
  // at all. They can also cause differences between the basic block trace
  // and the raw disassembled memtrace.
  if (isInstructionNoop(machine_instruction)) {
    return;
  }

  PerThreadData* current_shard_data =
      reinterpret_cast<PerThreadData*>(shard_data);
  current_shard_data->segment_cost_model->AddInstruction(machine_instruction);
}

}  // namespace latency_model
}  // namespace mlgo
