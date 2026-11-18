#ifndef COMPILER_OPT_MEMTRACE_COSTMODEL_INSTRUCTION_COUNTING_COSTMODEL_H_
#define COMPILER_OPT_MEMTRACE_COSTMODEL_INSTRUCTION_COUNTING_COSTMODEL_H_

#include "compiler_opt/memtrace_costmodel/costmodel.h"
#include "llvm/MC/MCInst.h"

namespace mlgo {
namespace latency_model {

class InstructionCountingCostModel : public CostModel {
 public:
  void AddInstruction(const llvm::MCInst& new_instruction) override;

  // This function does not actually get the cost of the instruction trace,
  // rather just serializing the instructions to a text file where they can
  // be inspected later.
  double GetCost() override;

 private:
  double total_cost_ = 0.0;
};

}  // namespace latency_model
}  // namespace mlgo

#endif  // COMPILER_OPT_MEMTRACE_COSTMODEL_INSTRUCTION_COUNTING_COSTMODEL_H_
