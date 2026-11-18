#ifndef COMPILER_OPT_MEMTRACE_COSTMODEL_ELF_METADATA_PARSER_H_
#define COMPILER_OPT_MEMTRACE_COSTMODEL_ELF_METADATA_PARSER_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"

namespace mlgo {
namespace latency_model {

// Utility that loads sections from an unstripped binary: linker build id, the
// symbols table and the bb address map (assuming -fbasic-block-address-map)
// Together, the latter 2 are used to capture the mbb address map for each
// function, identified by name (we also assume -fno-split-machine-functions).
// Because a lot can go wrong in loading the binary, we expose a factory method
// and hide the implementation.
class UnstrippedBinaryProcessor {
 public:
  std::string GetLinkerBuildID() const;

  // Information about a machine basic block (BB).
  struct BBInfo {
    // The address is the start address of the BB.
    uint64_t address = 0;
    // The size is the size of the BB.
    uint32_t size = 0;
    // The frequency is the number of times the BB was executed (profile
    // information).
    uint64_t frequency = 0;
  };

  struct FunctionBBInfo {
    explicit FunctionBBInfo(absl::string_view func_name, uint64_t func_address,
                            uint64_t func_entrycount, uint64_t section_index,
                            absl::Span<const BBInfo> bb_infos)
        : function_name(func_name),
          function_address(func_address),
          func_entrycount(func_entrycount),
          section_index(section_index),
          bb_infos(bb_infos) {}

    FunctionBBInfo() = delete;

    absl::string_view function_name;
    uint64_t function_address = 0;
    uint64_t func_entrycount = 0;
    uint64_t section_index = 0;
    absl::Span<const BBInfo> bb_infos;
  };

  // Call `record_processor` once, passing a vector containing, at index `i`,
  // the start binary address for machine basic block with ID `i`; and a vector
  // containing, at position 0, the function entrycount, and then at position
  // `i` the BB frequency of BB with ID `i - 1`. `record_processor` will be
  // called for each alias name separately.
  absl::Status ProcessBBAddrMap(
      absl::AnyInvocable<void(const FunctionBBInfo&)> record_processor) const;

  ~UnstrippedBinaryProcessor();

  static absl::StatusOr<std::unique_ptr<UnstrippedBinaryProcessor>> Create(
      absl::string_view unstripped_binary_path);

 private:
  std::unique_ptr<llvm::MemoryBuffer> buffer_;
  std::unique_ptr<llvm::object::ELFObjectFileBase> elfobj_;

  UnstrippedBinaryProcessor(
      std::unique_ptr<llvm::MemoryBuffer> buffer,
      std::unique_ptr<llvm::object::ELFObjectFileBase> elfobj);
};

}  // namespace latency_model
}  // namespace mlgo

#endif  // COMPILER_OPT_MEMTRACE_COSTMODEL_ELF_METADATA_PARSER_H_
