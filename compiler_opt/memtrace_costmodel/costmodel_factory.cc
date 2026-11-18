#include "compiler_opt/memtrace_costmodel/costmodel_factory.h"

#include <functional>
#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "compiler_opt/memtrace_costmodel/costmodel.h"
#include "compiler_opt/memtrace_costmodel/instruction_counting_costmodel.h"
#include "compiler_opt/memtrace_costmodel/print_costmodel.h"
#include "compiler_opt/memtrace_costmodel/trace_segment_mca.h"
#include "llvm/Target/TargetMachine.h"

namespace mlgo {
namespace latency_model {

std::string AbslUnparseFlag(CostModelType cost_model_type) {
  switch (cost_model_type) {
    case (CostModelType::Mca):
      return "mca";
    case (CostModelType::Print):
      return "print";
    case (CostModelType::InstructionCounting):
      return "instruction_counting";
    default:
      LOG(QFATAL) << "Cannot unparse cost model type "
                  << static_cast<int>(cost_model_type);
  }
}

bool AbslParseFlag(absl::string_view text, CostModelType* cost_model_type,
                   std::string* error) {
  if (text == "mca") {
    *cost_model_type = CostModelType::Mca;
    return true;
  } else if (text == "print") {
    *cost_model_type = CostModelType::Print;
    return true;
  } else if (text == "instruction_counting") {
    *cost_model_type = CostModelType::InstructionCounting;
    return true;
  } else {
    *error = "There is no cost model with the given name,";
    return false;
  }
}

void ValidateCostModelFlags(const CostModelType model_type,
                            absl::string_view gematria_model_path,
                            absl::string_view print_output_file) {
  if (model_type == CostModelType::Print && print_output_file.empty()) {
    LOG(QFATAL) << "--print_output_file must be specified when --model_type is "
                   "\"print\".";
  }
}

std::function<std::unique_ptr<CostModel>()> CreateCostModelFactory(
    const CostModelType model_type, llvm::TargetMachine* target_machine,
    absl::string_view llvm_cpu_name, absl::string_view target_triple,
    absl::string_view gematria_model_path, const int gematria_task_index,
    absl::string_view print_output_file) {
  if (model_type == CostModelType::Mca) {
    return [llvm_cpu_name = std::string(llvm_cpu_name),
            target_triple =
                std::string(target_triple)]() -> std::unique_ptr<CostModel> {
      std::unique_ptr<mlgo::latency_model::CostModel> cost_model =
          std::make_unique<mlgo::latency_model::TraceSegmentMca>(target_triple,
                                                                 llvm_cpu_name);
      return cost_model;
    };
  } else if (model_type == CostModelType::Print) {
    return [target_triple = std::string(target_triple),
            llvm_cpu_name = std::string(llvm_cpu_name),
            print_output_file = std::string(
                print_output_file)]() -> std::unique_ptr<CostModel> {
      std::unique_ptr<mlgo::latency_model::CostModel> cost_model =
          std::make_unique<mlgo::latency_model::PrintCostModel>(
              target_triple, llvm_cpu_name, print_output_file);
      return cost_model;
    };
  } else if (model_type == CostModelType::InstructionCounting) {
    return []() -> std::unique_ptr<CostModel> {
      std::unique_ptr<mlgo::latency_model::CostModel> cost_model =
          std::make_unique<InstructionCountingCostModel>();
      return cost_model;
    };
  } else {
    LOG(QFATAL) << "Unknown model type: " << static_cast<int>(model_type);
  }
}

}  // namespace latency_model
}  // namespace mlgo
