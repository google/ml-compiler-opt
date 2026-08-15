#include "compiler_opt/memtrace_costmodel/extract_corpus_subset_lib.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "compiler_opt/memtrace_costmodel/status_macros.h"
#include "llvm/ADT/StableHashing.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/StructuralHash.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SourceMgr.h"

namespace mlgo {
namespace latency_model {
namespace {

std::string GetFullModulePath(absl::string_view corpus_path,
                              absl::string_view module_relative_path) {
  return (std::filesystem::path(std::string(corpus_path)) /
          std::string(module_relative_path))
             .string() +
         ".bc";
}

// Contains information on an individual function relevant to corpus
// extraction. Designed to be relatively small so that it can efficiently
// be sent between threads after a bitcode module has been processed into
// a series of this struct.
struct FunctionInfo {
  uint32_t function_id = 0;
  size_t module_index = 0;
  llvm::stable_hash function_hash = 0;
  llvm::GlobalValue::LinkageTypes linkage_type =
      llvm::GlobalValue::LinkageTypes::ExternalLinkage;
};

// Checks if there are differing function definitions present, updating the
// function hashes mapping if necessary.
void CheckDifferingDefinitionsAndUpdateHashes(
    absl::flat_hash_map<uint32_t, llvm::stable_hash>& function_hashes,
    const FunctionInfo& current_function) {
  auto [function_hashes_it, inserted] = function_hashes.emplace(
      current_function.function_id, current_function.function_hash);
  if (!inserted) {
    if (function_hashes_it->second != current_function.function_hash) {
      // We can ignore weak/linkonce linkage types here as lld will always
      // resolve to a non-weak symbol, or the first weak symbol that is
      // found. So these symbols should never end up being used in a link
      // anyways, and thus we can safely skip them.
      if (current_function.linkage_type ==
              llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage ||
          current_function.linkage_type ==
              llvm::GlobalValue::LinkageTypes::WeakAnyLinkage) {
        LOG(WARNING) << "Found differing function definitions. Ignoring "
                        "due to weak linkage.\n";
        return;
      }
    }

    QCHECK(function_hashes_it->second == current_function.function_hash)
        << "Function " << current_function.function_id
        << " has multiple differing definitions.";
  }
}

std::vector<FunctionInfo> GetFunctionInfoForModule(
    const std::string& module_path, size_t module_index,
    const FunctionMapping& function_name_to_id) {
  llvm::LLVMContext llvm_context;

  auto memory_buffer_or = llvm::MemoryBuffer::getFile(module_path);
  QCHECK(memory_buffer_or) << "Failed to open module file: " << module_path;
  std::unique_ptr<llvm::MemoryBuffer> memory_buffer =
      std::move(*memory_buffer_or);
  llvm::MemoryBufferRef module_memory_buffer = memory_buffer->getMemBufferRef();

  llvm::SMDiagnostic parse_error;
  std::unique_ptr<llvm::Module> current_module =
      llvm::parseIR(module_memory_buffer, parse_error, llvm_context);
  QCHECK(current_module) << "Failed to parse IR for module: " << module_path
                         << " Error: " << parse_error.getMessage().str();

  std::vector<FunctionInfo> function_info;
  function_info.reserve(current_module->size());

  for (const llvm::Function& current_function : *current_module) {
    // Skip all declarations as we only care about the function definitions.
    if (current_function.isDeclaration()) {
      continue;
    }

    // If the function has the available_externally attribute, it will
    // not get emitted into the object file and thus we cannot put it
    // in the map as the BB trace modelling tooling works over object
    if (current_function.hasAvailableExternallyLinkage()) {
      continue;
    }

    const auto function_id_it =
        function_name_to_id.function_ids().find(current_function.getName());
    // The corpus will have more functions than the original binary, so if we
    // fail to find one in the name to ID map, we should simply skip it.
    // Missing functions will be caught later.
    if (function_id_it == function_name_to_id.function_ids().end()) {
      continue;
    }

    // TODO: The checks below use the standard StructuralHash
    // rather than the detailed StructuralHash due to issues with function
    // names being slightly different between definitions in different
    // modules. Eventually we should switch over to detailed StructuralHash
    // here.
    llvm::stable_hash function_hash =
        llvm::StructuralHash(current_function, false);

    FunctionInfo current_function_info = {
        .function_id = function_id_it->second,
        .module_index = module_index,
        .function_hash = function_hash,
        .linkage_type = current_function.getLinkage()};

    function_info.push_back(std::move(current_function_info));
  }

  // Additionally capture all global aliases as otherwise we might miss
  // functions where multiple symbols point to the same address, but we
  // only put one of them into function_to_module_name, leading to missing
  // functions. This is particularly common with C++ constructors and
  // destructors.
  for (const llvm::GlobalAlias& current_alias : current_module->aliases()) {
    if (const auto* aliasee_function =
            llvm::dyn_cast<llvm::Function>(current_alias.getAliaseeObject())) {
      QCHECK(!aliasee_function->isDeclaration());

      const auto function_id_it =
          function_name_to_id.function_ids().find(current_alias.getName());
      // The corpus will have more functions than the original binary, so if
      // we fail to find one in the name to ID map, we should simply skip it.
      // Missing functions will be caught later.
      if (function_id_it == function_name_to_id.function_ids().end()) {
        continue;
      }

      // We can just check for differing definitions for the alias name in the
      // same way as for functions. If the alias name does not have a function
      // ID, then it is not part of the trace and we only care that functions
      // with that exact alias name are identical because that is all we will
      // resolve. If it does have a function ID and there are multiple differing
      // definitions, that will be caught when iterating through the functions.

      // TODO: The checks below use the standard StructuralHash
      // rather than the detailed StructuralHash due to issues with function
      // names being slightly different between definitions in different
      // modules. Eventually we should switch over to detailed StructuralHash
      // here.
      llvm::stable_hash function_hash =
          llvm::StructuralHash(*aliasee_function, false);
      FunctionInfo current_function_info = {
          .function_id = function_id_it->second,
          .module_index = module_index,
          .function_hash = function_hash,
          .linkage_type = aliasee_function->getLinkage()};
      function_info.push_back(std::move(current_function_info));
    }
  }

  return function_info;
}

}  // end anonymous namespace

absl::flat_hash_map<uint32_t, std::string> GetFunctionToModuleMapping(
    const std::vector<std::string>& module_paths, absl::string_view corpus_path,
    const FunctionMapping& function_name_to_id) {
  absl::flat_hash_map<uint32_t, std::string> function_to_module_names;
  absl::flat_hash_map<uint32_t, llvm::stable_hash> function_hashes;

  for (size_t i = 0; i < module_paths.size(); ++i) {
    std::string module_full_path =
        GetFullModulePath(corpus_path, module_paths[i]);

    QCHECK(std::filesystem::exists(module_full_path))
        << "Module path does not exist: " << module_full_path;

    std::vector<FunctionInfo> current_module_info =
        GetFunctionInfoForModule(module_full_path, i, function_name_to_id);

    for (const FunctionInfo& current_function_info : current_module_info) {
      function_to_module_names.emplace(
          current_function_info.function_id,
          module_paths[current_function_info.module_index]);

      CheckDifferingDefinitionsAndUpdateHashes(function_hashes,
                                               current_function_info);
    }

    LOG(INFO) << "Finished processing module " << (i + 1) << "/"
              << module_paths.size() << "\n";
  }

  return function_to_module_names;
}

absl::flat_hash_set<std::string> GetIncludedFilesList(
    const absl::flat_hash_map<uint32_t, std::string>& function_to_module_names,
    const MbbTrace& mbb_trace) {
  absl::flat_hash_set<std::string> included_modules;

  for (const MachineBbId& bb_info : mbb_trace.mbbs()) {
    // Skip blocks that do not have function IDs, as they are from shared
    // object invocations and recorded and thus do not have a function name
    // associated with them.
    if (!bb_info.has_function_id()) {
      continue;
    }

    const auto function_to_module_it =
        function_to_module_names.find(bb_info.function_id());
    QCHECK(function_to_module_it != function_to_module_names.end())
        << "Failed to find a function, ID " << bb_info.function_id()
        << " in the corpus.";

    included_modules.emplace(function_to_module_it->second);
  }

  return included_modules;
}

absl::Status CopySubsetCorpus(
    absl::string_view input_corpus_base_path,
    absl::string_view output_corpus_base_path,
    const absl::flat_hash_set<std::string>& included_modules,
    nlohmann::json& corpus_description) {
  // See if the output path exists and if it does not, create it.
  std::error_code ec;
  std::filesystem::create_directories(std::string(output_corpus_base_path), ec);
  if (ec) {
    return absl::InternalError(
        absl::StrCat("Failed to create directory: ", output_corpus_base_path,
                     " Error: ", ec.message()));
  }

  // Copy the modules included in the corpus subset over to the output
  // directory.
  for (const std::string& included_module : included_modules) {
    std::string input_module_full_path =
        (std::filesystem::path(std::string(input_corpus_base_path)) /
         included_module)
            .string();
    std::string output_module_full_path =
        (std::filesystem::path(std::string(output_corpus_base_path)) /
         included_module)
            .string();

    // Create the output subdirectory if it does not exist.
    std::filesystem::create_directories(
        std::filesystem::path(output_module_full_path).parent_path(), ec);
    if (ec) {
      return absl::InternalError(absl::StrCat(
          "Failed to create directory: ",
          std::filesystem::path(output_module_full_path).parent_path().string(),
          " Error: ", ec.message()));
    }

    std::filesystem::copy_file(
        input_module_full_path + ".bc", output_module_full_path + ".bc",
        std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) {
      return absl::InternalError(
          absl::StrCat("Failed to copy file: ", input_module_full_path + ".bc",
                       " Error: ", ec.message()));
    }

    std::filesystem::copy_file(
        input_module_full_path + ".cmd", output_module_full_path + ".cmd",
        std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) {
      return absl::InternalError(
          absl::StrCat("Failed to copy file: ", input_module_full_path + ".cmd",
                       " Error: ", ec.message()));
    }

    // If we have a distributed ThinLTO corpus, we need to copy over the
    // ThinLTO index files (.thinlto.bc) to the output directory.
    if (corpus_description["has_thinlto"]) {
      std::filesystem::copy_file(
          input_module_full_path + ".thinlto.bc",
          output_module_full_path + ".thinlto.bc",
          std::filesystem::copy_options::overwrite_existing, ec);
      if (ec) {
        return absl::InternalError(absl::StrCat(
            "Failed to copy file: ", input_module_full_path + ".thinlto.bc",
            " Error: ", ec.message()));
      }
    }
  }

  // Modify the corpus description to only include the necessary files and
  // then copy them over to the output directory.
  corpus_description["modules"].clear();

  for (const std::string& included_module : included_modules) {
    corpus_description["modules"].push_back(included_module);
  }

  // Sort the output modules to ensure deterministic output of the corpus
  // description. This is mainly useful for testing purposes.
  std::sort(corpus_description["modules"].begin(),
            corpus_description["modules"].end());

  std::string output_corpus_description_path =
      (std::filesystem::path(std::string(output_corpus_base_path)) /
       "corpus_description.json")
          .string();

  // Write out the corpus description. Use an indent of two so the
  // corpus description is appropriately indented for manual inspection
  // and/or modification.
  std::ofstream output_file(output_corpus_description_path);
  if (!output_file) {
    return absl::InternalError(
        absl::StrCat("Failed to open output corpus description path: ",
                     output_corpus_description_path));
  }
  output_file << corpus_description.dump(2);
  if (!output_file.good()) {
    return absl::InternalError(
        absl::StrCat("Failed to write corpus description to path: ",
                     output_corpus_description_path));
  }

  return absl::OkStatus();
}

constexpr absl::string_view kFunctionSpecializationSentinel = ".specialized.";

FunctionMapping ProcessFunctionMappingForModifiedFunctions(
    const FunctionMapping& function_mapping) {
  FunctionMapping processed_function_mapping;

  for (const auto& [function_name, function_id] :
       function_mapping.function_ids()) {
    if (absl::StrContains(function_name, kFunctionSpecializationSentinel)) {
      absl::string_view standard_function_name =
          *absl::StrSplit(function_name, kFunctionSpecializationSentinel)
               .begin();
      processed_function_mapping.mutable_function_ids()->try_emplace(
          standard_function_name, function_id);
    }

    processed_function_mapping.mutable_function_ids()->try_emplace(
        function_name, function_id);
  }

  return processed_function_mapping;
}

}  // namespace latency_model
}  // namespace mlgo
