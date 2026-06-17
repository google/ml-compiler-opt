#include "compiler_opt/memtrace_costmodel/trace_segment_mca.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/MCA/Context.h"
#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/MCA/InstrBuilder.h"
#include "llvm/MCA/Instruction.h"
#include "llvm/MCA/Stages/Stage.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Triple.h"

constexpr const int kCallLatency = 100;

namespace mlgo {
namespace latency_model {

TraceSegmentMca::TraceSegmentMca(std::string_view target_triple,
                                 std::string_view cpu_name, int batch_size)
    : recycle_freed_instruction_(
          [this](llvm::mca::Instruction* freed_instruction) {
            recycled_mca_instructions_[&freed_instruction->getDesc()].insert(
                freed_instruction);
          }),
      get_recycled_instruction_(
          [this](const llvm::mca::InstrDesc& instruction_description)
              -> llvm::mca::Instruction* {
            auto recycled_instruction_iterator =
                recycled_mca_instructions_.find(&instruction_description);
            if (recycled_instruction_iterator !=
                recycled_mca_instructions_.end()) {
              llvm::SmallPtrSetImpl<llvm::mca::Instruction*>&
                  recycled_instructions = recycled_instruction_iterator->second;
              if (!recycled_instructions.empty()) {
                llvm::mca::Instruction* recycled_instruction =
                    *recycled_instructions.begin();
                recycled_instructions.erase(recycled_instruction);
                return recycled_instruction;
              }
            }
            return nullptr;
          }),
      batch_size_(batch_size) {
  std::string possible_lookup_error;
  llvm::Triple triple((llvm::StringRef(target_triple)));
  const llvm::Target* const llvm_target =
      llvm::TargetRegistry::lookupTarget(triple, possible_lookup_error);
  QCHECK(llvm_target) << possible_lookup_error;
  mc_subtarget_info_ = absl::WrapUnique(
      llvm_target->createMCSubtargetInfo(triple, cpu_name, ""));
  QCHECK(mc_subtarget_info_);
  mc_register_info_ = absl::WrapUnique(llvm_target->createMCRegInfo(triple));
  QCHECK(mc_register_info_);
  mc_instruction_info_ = absl::WrapUnique(llvm_target->createMCInstrInfo());
  QCHECK(mc_instruction_info_);
  mc_instruction_analysis_ = absl::WrapUnique(
      llvm_target->createMCInstrAnalysis(mc_instruction_info_.get()));
  QCHECK(mc_instruction_analysis_);
  mca_context_ = std::make_unique<llvm::mca::Context>(*mc_register_info_,
                                                      *mc_subtarget_info_);
  QCHECK(mca_context_);

  CreateMcaPipeline();

  mca_instrument_manager_ = std::make_unique<llvm::mca::InstrumentManager>(
      *mc_subtarget_info_, *mc_instruction_info_);
  QCHECK(mca_instrument_manager_);
  mca_instruction_builder_ = std::make_unique<llvm::mca::InstrBuilder>(
      *mc_subtarget_info_, *mc_instruction_info_, *mc_register_info_,
      mc_instruction_analysis_.get(), *mca_instrument_manager_, kCallLatency);
  QCHECK(mca_instruction_builder_);
  mca_instruction_processor_ =
      absl::WrapUnique(llvm_target->createInstrPostProcess(
          *mc_subtarget_info_, *mc_instruction_info_));
  // If there is not a target specific instruction post processor available,
  // create a generic one.
  if (!mca_instruction_processor_) {
    mca_instruction_processor_ =
        absl::WrapUnique(llvm_target->createInstrPostProcess(
            *mc_subtarget_info_, *mc_instruction_info_));
  }
  QCHECK(mca_instruction_processor_);

  source_manager_.setOnInstFreedCallback(recycle_freed_instruction_);
  mca_instruction_builder_->setInstRecycleCallback(get_recycled_instruction_);
}

void TraceSegmentMca::CreateMcaPipeline() {
  // Setting these values to zero/their defaults makes MCA use the values
  // provided by the scheduling model.
  llvm::mca::PipelineOptions mca_pipeline_options(
      /*UOPQSize=*/0, /*DecThr=*/0,
      /*DW=*/0,
      /*RFS=*/0,
      /*LQS=*/0, /*SQS=*/0,
      /*NoAlias=*/true,
      /*ShouldEnableBottleneckAnalysis=*/false);
  mca_custombehavior_ = std::make_unique<llvm::mca::CustomBehaviour>(
      *mc_subtarget_info_, source_manager_, *mc_instruction_info_);
  QCHECK(mca_custombehavior_);
  mca_pipeline_ = mca_context_->createDefaultPipeline(
      mca_pipeline_options, source_manager_, *mca_custombehavior_);
  QCHECK(mca_pipeline_);
}

void TraceSegmentMca::AddInstruction(const llvm::MCInst& new_instruction) {
  const llvm::MCInstrDesc& instruction_description =
      mc_instruction_info_->get(new_instruction.getOpcode());

  // Skip call and return instructions. These are not modeled properly by MCA
  // so omitting them will have little impact on modeling the performance
  // characteristics we are interested in. Passing in return instructions also
  // currently results in use after frees within MCA.
  if (instruction_description.isCall() || instruction_description.isReturn())
    return;

  llvm::Expected<std::unique_ptr<llvm::mca::Instruction>> new_mca_instruction =
      mca_instruction_builder_->createInstruction(new_instruction, /*IVec=*/{});

  if (!new_mca_instruction) {
    llvm::mca::Instruction* recycled_instruction = nullptr;
    llvm::Error leftover_error = llvm::handleErrors(
        new_mca_instruction.takeError(),
        [&recycled_instruction](
            const llvm::mca::RecycledInstErr& recycling_error) {
          recycled_instruction = recycling_error.getInst();
        });
    QCHECK(!leftover_error);
    QCHECK(recycled_instruction);
    llvm::consumeError(std::move(leftover_error));
    mca_instruction_processor_->postProcessInstruction(*recycled_instruction,
                                                       new_instruction);
    source_manager_.addRecycledInst(recycled_instruction);
  } else {
    // We only need to recycle instructions if they are new (i.e., not
    // recycled). Recycled instructions will already contain the relevant
    // modifications. The x86 instruction post processor is stateless, so this
    // is safe.
    mca_instruction_processor_->postProcessInstruction(
        *new_mca_instruction.get(), new_instruction);
    source_manager_.addInst(std::move(new_mca_instruction.get()));
  }

  ++instruction_count_;

  if (instruction_count_ % batch_size_ == 0) {
    llvm::Expected<unsigned> cycles = mca_pipeline_->run();

    // We expect to get an InstStreamPause error rather than a cycles
    // value as we have not called endOfStream yet. Check that this is
    // the case. Consume the error as we are expecting it, and we are
    // not done processing.
    QCHECK(!cycles);
    QCHECK(cycles.errorIsA<llvm::mca::InstStreamPause>());
    llvm::consumeError(cycles.takeError());
  }
}

double TraceSegmentMca::GetCost() {
  source_manager_.endOfStream();
  llvm::Expected<unsigned> cycles = mca_pipeline_->run();
  QCHECK(cycles);
  return cycles.get();
}

}  // namespace latency_model
}  // namespace mlgo
