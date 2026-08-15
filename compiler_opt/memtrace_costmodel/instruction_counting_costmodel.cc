#include "compiler_opt/memtrace_costmodel/instruction_counting_costmodel.h"

#include "llvm/MC/MCInst.h"

namespace mlgo {
namespace latency_model {

void InstructionCountingCostModel::AddInstruction(
    const llvm::MCInst& new_instruction) {
  total_cost_ += 1.0;
}

double InstructionCountingCostModel::GetCost() { return total_cost_; }

}  // namespace latency_model
}  // namespace mlgo
