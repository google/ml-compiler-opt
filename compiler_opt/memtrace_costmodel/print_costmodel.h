#ifndef COMPILER_OPT_MEMTRACE_COSTMODEL_PRINT_COSTMODEL_H_
#define COMPILER_OPT_MEMTRACE_COSTMODEL_PRINT_COSTMODEL_H_

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "compiler_opt/memtrace_costmodel/costmodel.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace mlgo {
namespace latency_model {

class PrintCostModel : public CostModel {
 public:
  explicit PrintCostModel(std::string_view target_triple,
                          std::string_view cpu_name,
                          std::string_view output_file_path);

  void AddInstruction(const llvm::MCInst& new_instruction) override;

  // This function does not actually get the cost of the instruction trace,
  // rather just serializing the instructions to a text file where they can
  // be inspected later.
  double GetCost() override;

 private:
  std::unique_ptr<llvm::MCInstrInfo> mc_instruction_info_;
  std::unique_ptr<llvm::MCRegisterInfo> mc_register_info_;
  std::unique_ptr<llvm::MCAsmInfo> mc_asm_info_;
  std::unique_ptr<llvm::MCInstPrinter> mc_instruction_printer_;
  std::unique_ptr<llvm::MCSubtargetInfo> mc_subtarget_info_;

  std::vector<std::string> instructions_;
  std::string output_file_path_;
};

}  // namespace latency_model
}  // namespace mlgo

#endif  // COMPILER_OPT_MEMTRACE_COSTMODEL_PRINT_COSTMODEL_H_
