#ifndef COMPILER_OPT_MEMTRACE_COSTMODEL_COSTMODEL_FACTORY_H_
#define COMPILER_OPT_MEMTRACE_COSTMODEL_COSTMODEL_FACTORY_H_

#include <functional>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "compiler_opt/memtrace_costmodel/costmodel.h"
#include "llvm/Target/TargetMachine.h"

namespace mlgo {
namespace latency_model {

enum class CostModelType {
  Invalid,
  Mca,
  Print,
  InstructionCounting,
  CacheLines
};

std::string AbslUnparseFlag(CostModelType cost_model_type);
bool AbslParseFlag(absl::string_view text, CostModelType* cost_model_type,
                   std::string* error);

// Validates all cost-model related flags, ensuring that the necessary
// parameters are set for the cost model that is requested.
void ValidateCostModelFlags(CostModelType model_type,
                            absl::string_view gematria_model_path,
                            absl::string_view print_output_file);

// Creates a factory function that will create the requested cost model
// with the provided flags.
std::function<std::unique_ptr<CostModel>()> CreateCostModelFactory(
    CostModelType model_type, llvm::TargetMachine* target_machine,
    absl::string_view llvm_cpu_name, absl::string_view target_triple,
    absl::string_view gematria_model_path, int gematria_task_index,
    absl::string_view print_output_file);

}  // namespace latency_model
}  // namespace mlgo

#endif  // COMPILER_OPT_MEMTRACE_COSTMODEL_COSTMODEL_FACTORY_H_
