#ifndef COMPILER_OPT_MEMTRACE_COSTMODEL_COSTMODEL_H_
#define COMPILER_OPT_MEMTRACE_COSTMODEL_COSTMODEL_H_

#include "llvm/MC/MCInst.h"

namespace mlgo {
namespace latency_model {

// This class is an abstraction around a cost model used to estimate the
// performance characteristics of a given sequence of instructions.
// It serves as a wrapper around the specific cost model implementation,
// providing a common interface for all cost models.
class CostModel {
 public:
  virtual ~CostModel() = default;

  // This function should be called for each instruction in a stream of
  // execution whose cost we are interested in modeling, such as individual
  // basic blocks or longer trace segments.
  virtual void AddInstruction(const llvm::MCInst& new_instruction) = 0;

  // This function should be called once all `AddInstruction` has been called
  // for all instructions in the stream being modeled. Returns the cost of the
  // instruction stream.
  virtual double GetCost() = 0;
};

}  // namespace latency_model
}  // namespace mlgo

#endif  // COMPILER_OPT_MEMTRACE_COSTMODEL_COSTMODEL_H_
