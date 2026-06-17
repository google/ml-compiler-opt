#ifndef COMPILER_OPT_MEMTRACE_COSTMODEL_TRACE_SEGMENT_MCA_H_
#define COMPILER_OPT_MEMTRACE_COSTMODEL_TRACE_SEGMENT_MCA_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <string_view>
#include <unordered_map>

#include "compiler_opt/memtrace_costmodel/costmodel.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MCA/Context.h"
#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/MCA/IncrementalSourceMgr.h"
#include "llvm/MCA/InstrBuilder.h"
#include "llvm/MCA/Instruction.h"
#include "llvm/MCA/Pipeline.h"

namespace mlgo {
namespace latency_model {

// This class is an abstraction around LLVM MCA (machine code analyzer), aimed
// specifically at supporting the memtrace case where instructions are
// processed one by one. This class automatically handles setting up the
// relevant LLVM State and keeping the pipeline state so users only have to
// add instructions one by one and then query for the cost after a trace
// segment is complete (i.e., upon entrypoint exit).
class TraceSegmentMca : public CostModel {
 public:
  explicit TraceSegmentMca(std::string_view target_triple,
                           std::string_view cpu_name, int batch_size = 1);

  // This function should be called for each instruction encountered during
  // memtrace processing within a context where we are interested in modeling
  // the cost (i.e., running under an entrypoint).
  void AddInstruction(const llvm::MCInst& new_instruction) override;

  // This function should only be called after all relevant instructions have
  // been added and the user is ready to get a final cost for a specific
  // instruction stream (i.e., upon entrypoint exit).
  double GetCost() override;

 private:
  void CreateMcaPipeline();

  std::unique_ptr<llvm::mca::Context> mca_context_;
  std::unique_ptr<llvm::MCSubtargetInfo> mc_subtarget_info_;
  std::unique_ptr<llvm::MCRegisterInfo> mc_register_info_;
  std::unique_ptr<llvm::MCInstrInfo> mc_instruction_info_;
  std::unique_ptr<llvm::MCInstrAnalysis> mc_instruction_analysis_;

  llvm::mca::IncrementalSourceMgr source_manager_;

  std::unique_ptr<llvm::mca::CustomBehaviour> mca_custombehavior_;
  std::unique_ptr<llvm::mca::Pipeline> mca_pipeline_;
  std::unique_ptr<llvm::mca::InstrumentManager> mca_instrument_manager_;
  std::unique_ptr<llvm::mca::InstrBuilder> mca_instruction_builder_;
  std::unique_ptr<llvm::mca::InstrPostProcess> mca_instruction_processor_;

  size_t instruction_count_ = 0;

  std::function<void(llvm::mca::Instruction*)> recycle_freed_instruction_;
  std::function<llvm::mca::Instruction*(const llvm::mca::InstrDesc&)>
      get_recycled_instruction_;

  std::unordered_map<const llvm::mca::InstrDesc*,
                     llvm::SmallPtrSet<llvm::mca::Instruction*, 2>>
      recycled_mca_instructions_;

  const int batch_size_;
};

}  // namespace latency_model
}  // namespace mlgo

#endif  // COMPILER_OPT_MEMTRACE_COSTMODEL_TRACE_SEGMENT_MCA_H_
