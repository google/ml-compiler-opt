#include "compiler_opt/memtrace_costmodel/elf_metadata_parser.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "compiler_opt/memtrace_costmodel/status_macros.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/BuildID.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"

namespace {

template <typename T>
absl::StatusOr<T> AsStatusOr(llvm::Expected<T>&& expected) {
  if (expected) {
    return std::move(*expected);
  }
  std::string ret;
  llvm::raw_string_ostream OS(ret);
  OS << expected.takeError();
  return absl::InternalError(ret);
}

template <typename V>
void InsertIntoVectorAtPos(std::vector<V>* vector, int index, V value) {
  if (vector->size() < index + 1) vector->resize(index + 1);
  (*vector)[index] = value;
}

}  // namespace

namespace mlgo {
namespace latency_model {

UnstrippedBinaryProcessor::UnstrippedBinaryProcessor(
    std::unique_ptr<llvm::MemoryBuffer> buffer,
    std::unique_ptr<llvm::object::ELFObjectFileBase> elfobj)
    : buffer_(std::move(buffer)), elfobj_(std::move(elfobj)) {}

UnstrippedBinaryProcessor::~UnstrippedBinaryProcessor() = default;

absl::StatusOr<std::unique_ptr<UnstrippedBinaryProcessor>>
UnstrippedBinaryProcessor::Create(absl::string_view unstripped_binary_path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer_or_err =
      llvm::MemoryBuffer::getFile(llvm::StringRef(
          unstripped_binary_path.data(), unstripped_binary_path.size()));
  if (std::error_code ec = buffer_or_err.getError()) {
    return absl::InternalError(
        absl::StrCat("Failed to open binary: ", ec.message()));
  }
  std::unique_ptr<llvm::MemoryBuffer> buffer = std::move(*buffer_or_err);

  llvm::Expected<std::unique_ptr<llvm::object::Binary>> obj_binary_or_err =
      llvm::object::createBinary(buffer->getMemBufferRef());
  if (!obj_binary_or_err) {
    std::string err_str;
    llvm::raw_string_ostream OS(err_str);
    OS << obj_binary_or_err.takeError();
    return absl::InternalError(err_str);
  }
  std::unique_ptr<llvm::object::Binary> obj_binary =
      std::move(*obj_binary_or_err);

  if (llvm::isa<llvm::object::ELFObjectFileBase>(obj_binary.get())) {
    auto elf_obj = std::unique_ptr<llvm::object::ELFObjectFileBase>(
        llvm::cast<llvm::object::ELFObjectFileBase>(obj_binary.release()));
    return absl::WrapUnique(
        new UnstrippedBinaryProcessor(std::move(buffer), std::move(elf_obj)));
  }
  return absl::InternalError(
      absl::StrCat("Non-elf binary: ", unstripped_binary_path));
}

std::string UnstrippedBinaryProcessor::GetLinkerBuildID() const {
  return ::llvm::toHex(::llvm::object::getBuildID(elfobj_.get()),
                       /*lowercase=*/true);
}

// NameInfo contains both a function name and the section that it is contained
// in so that can provide the user section information later on to disambiguate
// between functions in relocatable object files compiled with
// -ffunction-sections.
struct NameInfo {
  ::llvm::StringRef function_name;
  uint64_t function_section = 0;

  bool operator==(const NameInfo& other) const = default;
};

// We defer to sorting the function names as they are guaranteed to be unique
// and in many cases the vast majority of functions will all have the same
// section index.
bool operator<(const NameInfo& lhs, const NameInfo& rhs) {
  return lhs.function_name < rhs.function_name;
}

absl::Status UnstrippedBinaryProcessor::ProcessBBAddrMap(
    absl::AnyInvocable<void(const UnstrippedBinaryProcessor::FunctionBBInfo&)>
        record_processor) const {
  // Because of aliasing, an address would potentially be mapped to more than
  // one name. We'll report all that back as separate entries with different
  // names but same mapping. One of them should match what the compiler has.
  absl::flat_hash_map<uint64_t, ::llvm::SmallSet<NameInfo, 1>> address_to_name;
  for (const ::llvm::object::ELFSymbolRef& symbol : elfobj_->symbols()) {
    if (symbol.getSize() == 0) {
      continue;
    }

    ASSIGN_OR_RETURN(const auto symbol_type, AsStatusOr(symbol.getType()));

    if (symbol_type != ::llvm::object::SymbolRef::ST_Function) continue;
    ASSIGN_OR_RETURN(const auto address, AsStatusOr(symbol.getAddress()));

    // We want to skip symbols at address 0, but only in non-relocatable
    // binaries as real function symbols can exist at address 0 in relocatable
    // object files.
    if (address == 0 && !elfobj_->isRelocatableObject()) continue;
    ASSIGN_OR_RETURN(::llvm::StringRef name, AsStatusOr(symbol.getName()));

    ASSIGN_OR_RETURN(::llvm::object::section_iterator section_it,
                     AsStatusOr(symbol.getSection()));
    address_to_name[address].insert(
        {.function_name = name, .function_section = section_it->getIndex()});
  }
  std::vector<llvm::object::PGOAnalysisMap> pgo_data;
  ASSIGN_OR_RETURN(const auto bb_addr_map_list,
                   AsStatusOr(elfobj_->readBBAddrMap(std::nullopt, &pgo_data)));
  CHECK_EQ(pgo_data.size(), bb_addr_map_list.size());

  // Try to avoid churning through allocating/deallocating this buffer,
  // and instead, clear it whenever transitioning to a new function.
  std::vector<BBInfo> bb_infos;
  for (const auto& [bb_addr_map, pgo] : llvm::zip(bb_addr_map_list, pgo_data)) {
    CHECK_EQ(bb_addr_map.getBBEntries().size(), pgo.BBEntries.size());
    bb_infos.clear();
    auto name_iter = address_to_name.find(bb_addr_map.getFunctionAddress());
    if (name_iter == address_to_name.end()) {
      LOG(WARNING) << "Found a BB map entry without a symbol name: 0x"
                   << absl::Hex(bb_addr_map.getFunctionAddress());
      continue;
    }
    const auto& names = name_iter->second;
    for (const auto& [entry, freq] :
         llvm::zip(bb_addr_map.getBBEntries(), pgo.BBEntries)) {
      InsertIntoVectorAtPos(
          &bb_infos, entry.ID,
          {.address = entry.Offset + bb_addr_map.getFunctionAddress(),
           .size = entry.Size,
           .frequency = freq.BlockFreq.getFrequency()});
    }
    for (const auto& name : names) {
      FunctionBBInfo current_function_info(
          name.function_name, bb_addr_map.getFunctionAddress(),
          pgo.FuncEntryCount, name.function_section, bb_infos);
      record_processor(current_function_info);
    }
  }
  return absl::OkStatus();
}

}  // namespace latency_model
}  // namespace mlgo
