#include <cstdint>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "compiler_opt/memtrace_costmodel/basic_block_trace.h"
#include "llvm-c/Target.h"

ABSL_FLAG(std::string, memtrace_path, "",
          "The path to the memtrace to process.");
ABSL_FLAG(std::vector<std::string>, symbol_names, {},
          "The names of the entrypoint symbol.");
ABSL_FLAG(std::string, binary_path, "", "The path to the binary.");
ABSL_FLAG(std::string, output_folder, "", "The path to the output folder.");
ABSL_FLAG(bool, split_on_segment, false,
          "Whether or not to split entrypoints by segment");
ABSL_FLAG(int64_t, max_blocks_per_segment, 1 << 23,
          "The maximum number of blocks that can be in an individual segment.");

int main(int argc, char** argv) {
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86Disassembler();

  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  if (absl::GetFlag(FLAGS_memtrace_path).empty()) {
    LOG(QFATAL) << "--memtrace_path was not specified.";
  }

  if (absl::GetFlag(FLAGS_symbol_names).empty()) {
    LOG(QFATAL) << "--symbol_names must be specified.";
  }

  if (absl::GetFlag(FLAGS_binary_path).empty()) {
    LOG(QFATAL) << "--binary_path was not specified.";
  }

  if (absl::GetFlag(FLAGS_output_folder).empty()) {
    LOG(QFATAL) << "--output_folder was not specified.";
  }

  QCHECK_OK(mlgo::latency_model::WriteBasicBlockTraces(
      absl::GetFlag(FLAGS_memtrace_path), absl::GetFlag(FLAGS_binary_path),
      absl::GetFlag(FLAGS_output_folder), absl::GetFlag(FLAGS_symbol_names),
      absl::GetFlag(FLAGS_split_on_segment),
      absl::GetFlag(FLAGS_max_blocks_per_segment)));

  return 0;
}
