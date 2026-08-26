#include "compiler_opt/memtrace_costmodel/print_costmodel.h"

#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

namespace mlgo {
namespace latency_model {

PrintCostModel::PrintCostModel(std::string_view target_triple,
                               std::string_view cpu_name,
                               std::string_view output_file_path)
    : output_file_path_(output_file_path) {
  std::string possible_lookup_error;
  llvm::Triple triple(target_triple);
  const llvm::Target* const llvm_target =
      llvm::TargetRegistry::lookupTarget(triple, possible_lookup_error);
  QCHECK(llvm_target);

  mc_instruction_info_ = absl::WrapUnique(llvm_target->createMCInstrInfo());
  QCHECK(mc_instruction_info_);

  mc_register_info_ = absl::WrapUnique(llvm_target->createMCRegInfo(triple));
  QCHECK(mc_register_info_);

  llvm::MCTargetOptions target_options;
  mc_asm_info_ = absl::WrapUnique(
      llvm_target->createMCAsmInfo(*mc_register_info_, triple, target_options));
  QCHECK(mc_asm_info_);

  mc_instruction_printer_ = absl::WrapUnique(llvm_target->createMCInstPrinter(
      llvm::Triple(target_triple), llvm::InlineAsm::AD_ATT, *mc_asm_info_,
      *mc_instruction_info_, *mc_register_info_));

  mc_subtarget_info_ = absl::WrapUnique(
      llvm_target->createMCSubtargetInfo(triple, cpu_name, ""));
}

void PrintCostModel::AddInstruction(const llvm::MCInst& new_instruction) {
  std::string output_buffer;
  llvm::raw_string_ostream output_stream(output_buffer);
  mc_instruction_printer_->printInst(&new_instruction, 0, "",
                                     *mc_subtarget_info_, output_stream);
  instructions_.push_back(std::move(output_buffer));
}

double PrintCostModel::GetCost() {
  if (!instructions_.empty()) {
    std::string output_buffer;
    llvm::raw_string_ostream output_stream(output_buffer);

    for (const std::string_view instruction : instructions_) {
      output_stream << instruction << "\n";
    }

    std::ofstream output_file(output_file_path_);
    QCHECK(output_file) << "Failed to open print costmodel output file: "
                        << output_file_path_;
    output_file << output_buffer;
    QCHECK(output_file.good())
        << "Failed to write to output file: " << output_file_path_;
  }

  return 0;
}

}  // namespace latency_model
}  // namespace mlgo
