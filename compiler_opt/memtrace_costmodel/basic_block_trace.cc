#include "compiler_opt/memtrace_costmodel/basic_block_trace.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "compiler_opt/memtrace_costmodel/elf_metadata_parser.h"
#include "compiler_opt/memtrace_costmodel/mbb_trace.proto.h"
#include "compiler_opt/memtrace_costmodel/serialized_mbbs.proto.h"
#include "drmemtrace/analyzer.h"
#undef X86
#undef X86_64
#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "compiler_opt/memtrace_costmodel/status_macros.h"
#include "drmemtrace/memref.h"
#include "drmemtrace/trace_entry.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "riegeli/bytes/file_writer.h"
#include "riegeli/records/record_writer.h"

namespace mlgo {
namespace latency_model {
namespace {
constexpr int kBbTraceCompressionLevel = 1 << 16;
constexpr absl::string_view kBbTraceFileName = "bb_trace.pb";
constexpr absl::string_view kFunctionIndexFileName = "function_index.pb";

template <typename T>
bool LlvmExpectedSucceeded(llvm::Expected<T>& expected) {
  return static_cast<bool>(expected);
}

std::vector<SectionInfo> GetSectionsWithBbInfoFromBinary(
    absl::string_view binary_path,
    const absl::flat_hash_map<uint64_t, MachineBbId>&
        bb_addresses_to_ids_and_functions) {
  llvm::Expected<llvm::object::OwningBinary<llvm::object::Binary>>
      application_binary = llvm::object::createBinary(binary_path);

  QCHECK(LlvmExpectedSucceeded(application_binary));

  llvm::object::ELFObjectFileBase* object_file =
      llvm::cast<llvm::object::ELFObjectFileBase>(
          application_binary->getBinary());

  std::vector<SectionInfo> sections_with_bb_info;

  for (const llvm::object::SectionRef& section : object_file->sections()) {
    if (!section.isText()) {
      continue;
    }

    uint64_t section_address = section.getAddress();
    uint64_t section_size = section.getSize();

    bool contains_bb_address = false;
    for (const auto& [bb_address, _] : bb_addresses_to_ids_and_functions) {
      if (bb_address >= section_address &&
          bb_address < section_address + section_size) {
        contains_bb_address = true;
        break;
      }
    }

    if (contains_bb_address) {
      sections_with_bb_info.push_back(
          {.address = section_address, .size = section_size});
    }
  }

  return sections_with_bb_info;
}

bool InsideTextSectionWithBbAddrMap(std::vector<SectionInfo>& section_info,
                                    uint64_t current_address) {
  for (const SectionInfo& current_section : section_info) {
    if (current_section.address <= current_address &&
        current_section.address + current_section.size > current_address) {
      return true;
    }
  }
  return false;
}

llvm::MCInst getInstructionFromBytes(llvm::ArrayRef<uint8_t> instruction_data,
                                     llvm::MCDisassembler& disassembler,
                                     uint64_t& instruction_size,
                                     uint64_t instruction_address = 0) {
  llvm::MCInst instruction;
  std::string disassembler_output_buffer;
  llvm::raw_string_ostream output_stream(disassembler_output_buffer);

  const llvm::MCDisassembler::DecodeStatus status = disassembler.getInstruction(
      instruction, instruction_size, instruction_data, instruction_address,
      output_stream);
  output_stream.flush();
  if (status != llvm::MCDisassembler::DecodeStatus::Success) {
    LOG(WARNING) << "Failed to disassemble instruction at "
                 << instruction_address << " status: " << status
                 << " output: " << disassembler_output_buffer;
    return llvm::MCInst();
  }
  return instruction;
}
}  // namespace

uint64_t GetElfOffset(
    const dynamorio::drmemtrace::module_mapper_t* module_mapper,
    uint64_t runtime_pc) {
  const auto& modules =
      const_cast<dynamorio::drmemtrace::module_mapper_t*>(module_mapper)
          ->get_loaded_modules();
  for (const auto& m : modules) {
    if (runtime_pc >= reinterpret_cast<uint64_t>(m.orig_seg_base) &&
        runtime_pc < reinterpret_cast<uint64_t>(m.orig_seg_base) + m.seg_size) {
      return runtime_pc - reinterpret_cast<uint64_t>(m.orig_seg_base) +
             m.seg_offs;
    }
  }
  return 0;  // not found
}

void* BBMemtraceProcessor::parallel_shard_init(int shard_index,
                                               void* worker_data) {
  PerShardData* current_shard_data = new PerShardData();
  LOG(INFO) << "Shard init " << shard_index << " data: " << current_shard_data;
  return reinterpret_cast<void*>(current_shard_data);
}

bool BBMemtraceProcessor::parallel_shard_exit(void* shard_data) {
  PerShardData* current_shard_data =
      reinterpret_cast<PerShardData*>(shard_data);

  {
    absl::MutexLock lock(&current_shard_data->mutex);
    if (!current_shard_data->current_trace_data.mbbs().empty() ||
        !current_shard_data->current_trace_data.shared_object_traces()
             .empty()) {
      absl::MutexLock lock(&bb_trace_processor_mutex_);
      bb_trace_processor_(current_shard_data->current_trace_data);
    }

    {
      absl::MutexLock function_name_to_id_lock(&function_name_to_id_mutex_);
      function_name_to_id_.mutable_function_ids()->insert(
          current_shard_data->function_name_to_id.begin(),
          current_shard_data->function_name_to_id.end());
    }
  }

  delete current_shard_data;
  return true;
}

bool BBMemtraceProcessor::process_memref(
    const dynamorio::drmemtrace::memref_t& memref) {
  LOG(QFATAL) << "Intentionally not implemented";
  return false;
}

bool BBMemtraceProcessor::parallel_shard_memref(
    void* shard_data, const dynamorio::drmemtrace::memref_t& memref) {
  PerShardData* current_shard_data =
      reinterpret_cast<PerShardData*>(shard_data);
  absl::MutexLock lock(&current_shard_data->mutex);

  if (!dynamorio::drmemtrace::type_is_instr(memref.instr.type)) return true;

  // Size-based segment splitting
  if (!current_shard_data->inside_shared_object &&
      current_shard_data->current_trace_data.mbbs_size() >=
          max_blocks_per_segment_) {
    {
      absl::MutexLock lock(bb_trace_processor_mutex_);
      bb_trace_processor_(current_shard_data->current_trace_data);
    }
    current_shard_data->current_trace_data.Clear();
  }

  // Resolve ELF address using custom module mapping
  uint64_t elf_pc = GetElfOffset(module_mapper_, memref.instr.addr);
  if (elf_pc == 0) return true;  // Ignore unmapped code

  // Handle shared object boundary crossing
  if (!current_shard_data->inside_shared_object &&
      !InsideTextSectionWithBbAddrMap(sections_with_bb_info_, elf_pc)) {
    current_shard_data->inside_shared_object = true;

    // We have just entered the shared object. Create a new shared object trace.
    current_shard_data->current_trace_data.add_shared_object_traces();

    {
      absl::MutexLock current_trace_id_lock(current_trace_id_mutex_);
      current_shard_data->current_trace_data.mutable_shared_object_traces()
          ->rbegin()
          ->set_trace_id(current_trace_id_);
      ++current_trace_id_;
    }
  }

  if (current_shard_data->inside_shared_object &&
      InsideTextSectionWithBbAddrMap(sections_with_bb_info_, elf_pc)) {
    current_shard_data->inside_shared_object = false;

    // Left shared object, create a block referencing the trace
    MachineBbId* trace_bb_id =
        current_shard_data->current_trace_data.add_mbbs();
    trace_bb_id->set_basic_block_id(
        current_shard_data->current_trace_data.shared_object_traces()
            .rbegin()
            ->trace_id());
  }

  if (current_shard_data->inside_shared_object) {
    std::string current_instruction_data(
        reinterpret_cast<const char*>(memref.instr.encoding),
        memref.instr.size);

    current_shard_data->current_trace_data.mutable_shared_object_traces()
        ->rbegin()
        ->add_instruction_data(std::move(current_instruction_data));
  }

  auto function_address_and_bb_id =
      bb_addresses_to_ids_and_functions_.find(elf_pc);
  if (function_address_and_bb_id != bb_addresses_to_ids_and_functions_.end()) {
    MachineBbId* next_mbb = current_shard_data->current_trace_data.add_mbbs();
    (*next_mbb) = function_address_and_bb_id->second;

    current_shard_data->function_name_to_id.try_emplace(
        function_id_to_name_[next_mbb->function_id()], next_mbb->function_id());
  }

  return true;
}

absl::Status GetBasicBlockTracesFromDirectory(
    absl::string_view trace_dir, absl::string_view binary_path,
    absl::FunctionRef<void(const MbbTrace&)> bb_trace_processor,
    absl::Span<const uint64_t> symbols_of_interest_addresses,
    FunctionMapping& function_name_to_id, bool split_on_segment,
    int64_t max_blocks_per_segment) {
  std::vector<std::string> function_id_to_name;

  ASSIGN_OR_RETURN(auto bb_addresses_to_ids_and_functions,
                   GetBbAddressesToIdsMap(binary_path, function_id_to_name));

  std::vector<SectionInfo> section_info = GetSectionsWithBbInfoFromBinary(
      binary_path, bb_addresses_to_ids_and_functions);

  ASSIGN_OR_RETURN(
      const auto unstripped_binary_processor,
      mlgo::latency_model::UnstrippedBinaryProcessor::Create(binary_path));

  // Read modules.log file contents
  std::string modules_log_path = std::string(trace_dir) + "/modules.log";
  std::ifstream modules_file(modules_log_path,
                             std::ios::binary | std::ios::ate);
  if (!modules_file) {
    return absl::InternalError(
        absl::StrCat("Failed to open modules.log at ", modules_log_path));
  }
  std::streamsize size = modules_file.tellg();
  modules_file.seekg(0, std::ios::beg);
  std::vector<char> modules_buffer(size + 1, 0);
  if (!modules_file.read(modules_buffer.data(), size)) {
    return absl::InternalError("Failed to read modules.log");
  }

  // Create module mapper
  std::unique_ptr<dynamorio::drmemtrace::module_mapper_t> module_mapper =
      dynamorio::drmemtrace::module_mapper_t::create(modules_buffer.data());
  if (!module_mapper || !module_mapper->get_last_error().empty()) {
    return absl::InternalError(
        absl::StrCat("Failed to create module mapper: ",
                     module_mapper ? module_mapper->get_last_error() : ""));
  }

  std::vector<std::unique_ptr<dynamorio::drmemtrace::analysis_tool_t>> tools;
  tools.push_back(std::make_unique<BBMemtraceProcessor>(
      bb_addresses_to_ids_and_functions, function_id_to_name,
      module_mapper.get(), symbols_of_interest_addresses, bb_trace_processor,
      function_name_to_id, split_on_segment, max_blocks_per_segment,
      std::move(section_info), unstripped_binary_processor->GetLinkerBuildID(),
      std::string(binary_path)));

  std::vector<dynamorio::drmemtrace::analysis_tool_t*> tool_ptrs;
  tool_ptrs.reserve(tools.size());
  for (const auto& t : tools) {
    tool_ptrs.push_back(t.get());
  }

  // Run standard DynamoRIO trace analyzer
  dynamorio::drmemtrace::analyzer_t analyzer(
      std::string(trace_dir), tool_ptrs.data(), tool_ptrs.size());
  if (!analyzer) {
    return absl::InternalError("Failed to initialize trace analyzer");
  }
  if (!analyzer.run()) {
    return absl::InternalError("Failed to run trace analyzer");
  }

  return absl::OkStatus();
}

absl::Status GetBasicBlockTracesFromDirectory(
    absl::string_view trace_dir, absl::string_view binary_path,
    absl::FunctionRef<void(const MbbTrace&)> bb_trace_processor,
    absl::Span<const std::string> symbols_of_interest,
    FunctionMapping& function_name_to_id, bool split_on_segment,
    int64_t max_blocks_per_segment) {
  ASSIGN_OR_RETURN(
      const auto unstripped_binary_processor,
      mlgo::latency_model::UnstrippedBinaryProcessor::Create(binary_path));
  std::vector<uint64_t> entrypoint_addresses;
  RETURN_IF_ERROR(unstripped_binary_processor->ProcessBBAddrMap(
      [&symbols_of_interest, &entrypoint_addresses](
          const UnstrippedBinaryProcessor::FunctionBBInfo& function_bb_info) {
        // We simply iterate over the symbols as there will always be very
        // few (n<5), and moving the symbols to a set and hashing would likely
        // be more expensive.
        for (absl::string_view symbol_of_interest : symbols_of_interest) {
          if (symbol_of_interest != function_bb_info.function_name) {
            continue;
          }
          entrypoint_addresses.push_back(function_bb_info.function_address);
        }
      }));
  CHECK_EQ(symbols_of_interest.size(), entrypoint_addresses.size())
      << "Expected to find only one address per entrypoint.";

  return GetBasicBlockTracesFromDirectory(
      trace_dir, binary_path, bb_trace_processor, entrypoint_addresses,
      function_name_to_id, split_on_segment, max_blocks_per_segment);
}

absl::StatusOr<absl::flat_hash_map<uint64_t, MachineBbId>>
GetBbAddressesToIdsMap(absl::string_view binary_path,
                       std::vector<std::string>& function_id_to_name) {
  ASSIGN_OR_RETURN(
      const auto unstripped_binary_processor,
      mlgo::latency_model::UnstrippedBinaryProcessor::Create(binary_path));

  BinaryApplicationToBbDisassembler bb_disassembler("x86_64",
                                                    std::string(binary_path));

  absl::flat_hash_map<uint64_t, MachineBbId> bb_addresses_to_ids_and_functions;
  absl::flat_hash_map<std::string, uint32_t> function_name_to_id_map;

  RETURN_IF_ERROR(unstripped_binary_processor->ProcessBBAddrMap(
      [&bb_addresses_to_ids_and_functions, &function_name_to_id_map,
       &function_id_to_name, &bb_disassembler](
          const UnstrippedBinaryProcessor::FunctionBBInfo& function_bb_info) {
        for (uint32_t i = 0; i < function_bb_info.bb_infos.size(); ++i) {
          if (function_bb_info.bb_infos[i].address == 0) continue;

          // Skip empty basic blocks as if there is another block at exactly
          // the same address that has a non-zero size, we might end up picking
          // up the zero sized BB which would cause downstream consumers to
          // use the wrong instructions.
          if (function_bb_info.bb_infos[i].size == 0) continue;

          // TODO: For now, assert if we find .cold functions as
          // we need to ensure that we can handle them.
          QCHECK(!function_bb_info.function_name.ends_with(".cold"));

          const auto function_id_it =
              function_name_to_id_map.find(function_bb_info.function_name);
          uint32_t function_id = 0;
          if (function_id_it == function_name_to_id_map.end()) {
            function_id = function_name_to_id_map.size();
            function_name_to_id_map.emplace(function_bb_info.function_name,
                                            function_id);

            function_id_to_name.push_back(
                std::string(function_bb_info.function_name));
            QCHECK(function_id_to_name.size() ==
                   function_name_to_id_map.size());
          } else {
            function_id = function_id_it->second;
          }

          uint32_t current_entry = 0;

          bb_disassembler.ProcessAllEntriesInBlock(
              function_bb_info.bb_infos[i].address,
              function_bb_info.bb_infos[i].size,
              [&](uint64_t address_offset,
                  std::vector<InstructionInfo> entry_instructions,
                  llvm::ArrayRef<uint8_t> entry_contents) -> void {
                MachineBbId function_bb_id;
                function_bb_id.set_function_id(function_id);
                function_bb_id.set_basic_block_id(i);
                function_bb_id.set_entry_id(current_entry++);
                auto [_, inserted_bb] =
                    bb_addresses_to_ids_and_functions.emplace(
                        function_bb_info.bb_infos[i].address + address_offset,
                        function_bb_id);

                // Assert that if a BB is not inserted, the basic block IDs
                // match up. If a BB is not inserted, that means there is
                // another BB already present at that address, which leaves it
                // ambiguous which one should be used for modelling. We cannot
                // assert that the function name matches as there are multiple
                // symbol names that point to the same definition in some cases,
                // like whole object and base object constructors and
                // destructors.
                if (!inserted_bb) {
                  QCHECK_EQ(i, bb_addresses_to_ids_and_functions
                                   [function_bb_info.bb_infos[i].address]
                                       .basic_block_id());
                }
              });
        }
      }));

  CHECK_GT(bb_addresses_to_ids_and_functions.size(), 0);
  return bb_addresses_to_ids_and_functions;
}

absl::Status WriteBasicBlockTraces(
    absl::string_view trace_dir, absl::string_view binary_path,
    absl::string_view output_folder,
    absl::Span<const std::string> symbols_of_interest, bool split_on_segment,
    int64_t max_blocks_per_segment) {
  riegeli::RecordWriter output_writer(
      riegeli::Maker<riegeli::FileWriter>(std::string(output_folder) + "/" +
                                          std::string(kBbTraceFileName)),
      riegeli::RecordWriterBase::Options().set_zstd());

  auto bb_trace_processor =
      [&output_writer](const mlgo::latency_model::MbbTrace& mbb_trace_segment) {
        LOG(INFO) << "Writing a trace with " << mbb_trace_segment.mbbs_size()
                  << " BBs";
        output_writer.WriteRecord(mbb_trace_segment);
      };

  mlgo::latency_model::FunctionMapping function_name_to_id;
  RETURN_IF_ERROR(GetBasicBlockTracesFromDirectory(
      trace_dir, binary_path, bb_trace_processor, symbols_of_interest,
      function_name_to_id, split_on_segment, max_blocks_per_segment));
  QCHECK(output_writer.Close()) << output_writer.status();

  if (function_name_to_id.function_ids().empty()) {
    LOG(WARNING)
        << "No traces were written to "
        << std::string(output_folder) + "/" + std::string(kBbTraceFileName)
        << ". Check if symbols of interest are present in the profile.";
  }

  riegeli::RecordWriter function_index_writer(
      riegeli::Maker<riegeli::FileWriter>(std::string(output_folder) + "/" +
                                          std::string(kFunctionIndexFileName)),
      riegeli::RecordWriterBase::Options().set_zstd());
  function_index_writer.WriteRecord(function_name_to_id);
  QCHECK(function_index_writer.Close()) << function_index_writer.status();
  return absl::OkStatus();
}

ApplicationToBbDisassembler::~ApplicationToBbDisassembler() = default;

void ApplicationToBbDisassembler::PopulateLlvmHelpers(
    const std::string& target_triple) {
  std::string possible_lookup_error;
  llvm::Triple triple(target_triple);
  const llvm::Target* const llvm_target =
      llvm::TargetRegistry::lookupTarget(triple, possible_lookup_error);
  QCHECK(llvm_target);

  llvm::TargetOptions llvm_target_options;

  llvm_target_machine_.reset(llvm_target->createTargetMachine(
      triple, /*CPU*/ "", /*Features*/ "", llvm_target_options, std::nullopt));
  QCHECK(llvm_target_machine_);

  llvm_mc_context_ = std::make_unique<llvm::MCContext>(
      llvm_target_machine_->getTargetTriple(),
      llvm_target_machine_->getMCAsmInfo(),
      llvm_target_machine_->getMCRegisterInfo(),
      llvm_target_machine_->getMCSubtargetInfo());
  QCHECK(llvm_mc_context_);

  llvm_mc_disassembler_.reset(llvm_target->createMCDisassembler(
      llvm_target_machine_->getMCSubtargetInfo(), *llvm_mc_context_));
  QCHECK(llvm_mc_disassembler_);

  llvm_mc_instr_info_.reset(llvm_target->createMCInstrInfo());
  QCHECK(llvm_mc_instr_info_);
}

ApplicationToBbDisassembler::ApplicationToBbDisassembler(
    const std::string& target_triple) {
  PopulateLlvmHelpers(target_triple);
}

llvm::ArrayRef<InstructionInfo>
ApplicationToBbDisassembler::GetDisassembledInstructions(
    MachineBbId function_basic_block_id) {
  // TODO: This hurts performance. We should remove it once we
  // can statically know about all the BB traces at the beginning and add
  // them to the cache at that point.
  absl::ReaderMutexLock disassembling_lock(disassembling_instructions_mutex_);

  const auto disassembled_basic_block =
      disassembled_instructions_.find(function_basic_block_id);

  if (disassembled_basic_block == disassembled_instructions_.end() &&
      function_basic_block_id.entry_id() > 0) {
    return {};
  }

  QCHECK(disassembled_basic_block != disassembled_instructions_.end())
      << function_basic_block_id.has_function_id() << ":"
      << function_basic_block_id.function_id() << ":"
      << function_basic_block_id.basic_block_id() << ":"
      << function_basic_block_id.entry_id() << "\n";

  return disassembled_basic_block->second;
}

void ApplicationToBbDisassembler::ProcessAllEntriesFromBlockContents(
    uint64_t block_start_address,
    absl::FunctionRef<void(uint64_t, std::vector<InstructionInfo>&&,
                           llvm::ArrayRef<uint8_t>)>
        entry_processor,
    llvm::ArrayRef<uint8_t> block_contents,
    uint64_t expected_block_size) const {
  if (block_contents.empty()) {
    entry_processor(0, {}, {});
    return;
  }

  size_t previous_offset = 0;
  size_t current_offset = 0;
  std::vector<InstructionInfo> bb_instructions;
  while (current_offset < expected_block_size &&
         current_offset < block_contents.size()) {
    // while (current_offset < block_contents.size()) {
    uint64_t current_instruction_size = 0;

    const uint8_t* instruction_bytes_address =
        &block_contents.data()[current_offset];
    llvm::ArrayRef<uint8_t> instruction_data(
        instruction_bytes_address, block_contents.size() - current_offset);
    llvm::MCInst current_instruction = getInstructionFromBytes(
        instruction_data, *llvm_mc_disassembler_, current_instruction_size);

    const llvm::MCInstrDesc& current_instruction_description =
        llvm_mc_instr_info_->get(current_instruction.getOpcode());

    bb_instructions.push_back({.instruction = std::move(current_instruction),
                               .address = block_start_address + current_offset,
                               .size = current_instruction_size});
    current_offset += current_instruction_size;

    QCHECK_NE(current_instruction_size, 0);

    // We should always be before the end of the block or exactly at the end
    // of the block.
    QCHECK(current_offset <= block_contents.size());

    // We cannot have an entry at the end of the block because that is
    // actually just the first entry for the next block, so if we run into
    // this case, just return here.
    if (current_offset == block_contents.size()) {
      llvm::ArrayRef<uint8_t> entry_contents(
          block_contents.data() + previous_offset,
          current_offset - previous_offset);
      entry_processor(previous_offset, std::move(bb_instructions),
                      entry_contents);
      return;
    }

    // Split around calls as they represent a change in control flow not
    // captured by the compiler's definition of a basic block.
    // Additionally split around any terminator instructions to handle cases
    // like inline assembly where a terminator instruction such as a jump
    // might be placed in the middle of a block.
    if (current_instruction_description.isCall() ||
        current_instruction_description.isTerminator()) {
      llvm::ArrayRef<uint8_t> entry_contents(
          block_contents.data() + previous_offset,
          current_offset - previous_offset);
      entry_processor(previous_offset, std::move(bb_instructions),
                      entry_contents);
      bb_instructions.clear();
      previous_offset = current_offset;
    }
  }
}

void ApplicationToBbDisassembler::LoadSharedObjectTraces(
    const MbbTrace& mbb_trace) {
  absl::MutexLock disassembling_lock(disassembling_instructions_mutex_);

  for (const SharedObjectTrace& shared_object_trace :
       mbb_trace.shared_object_traces()) {
    std::vector<InstructionInfo> trace_instructions;
    trace_instructions.reserve(shared_object_trace.instruction_data_size());

    for (const std::string& instruction_encoding :
         shared_object_trace.instruction_data()) {
      uint64_t total_encoding_size = 0;
      while (total_encoding_size < instruction_encoding.size()) {
        uint64_t instruction_size;
        llvm::ArrayRef<uint8_t> instruction_data(
            reinterpret_cast<const uint8_t*>(instruction_encoding.data() +
                                             total_encoding_size),
            instruction_encoding.size());

        trace_instructions.push_back(
            {.instruction = getInstructionFromBytes(
                 instruction_data, *llvm_mc_disassembler_, instruction_size),
             .address = 0,
             .size = instruction_size});

        total_encoding_size += instruction_size;
      }
    }

    MachineBbId mbb_to_insert;
    mbb_to_insert.set_basic_block_id(shared_object_trace.trace_id());
    disassembled_instructions_[mbb_to_insert] = std::move(trace_instructions);
  }
}

void ApplicationToBbDisassembler::LoadSerializedBbs(
    const SerializedMbbs& serialized_mbbs) {
  absl::MutexLock disassembling_lock(disassembling_instructions_mutex_);

  for (const auto& [mbb_id, mbb_data] :
       llvm::zip(serialized_mbbs.mbb_ids(), serialized_mbbs.mbb_bytes())) {
    uint64_t current_offset = 0;
    std::vector<InstructionInfo> current_block_instructions;

    while (current_offset < mbb_data.size()) {
      llvm::ArrayRef<uint8_t> instruction_data(
          reinterpret_cast<const uint8_t*>(mbb_data.data() + current_offset),
          mbb_data.size() - current_offset);

      uint64_t instruction_size = 0;
      current_block_instructions.push_back(
          {.instruction = getInstructionFromBytes(
               instruction_data, *llvm_mc_disassembler_, instruction_size),
           .address = 0,
           .size = instruction_size});

      current_offset += instruction_size;
    }

    disassembled_instructions_.emplace(mbb_id,
                                       std::move(current_block_instructions));
  }
}

void BinaryApplicationToBbDisassembler::ProcessAllEntriesInBlock(
    uint64_t block_address, uint32_t block_size,
    absl::FunctionRef<void(uint64_t, std::vector<InstructionInfo>,
                           llvm::ArrayRef<uint8_t>)>
        entry_processor) const {
  llvm::ArrayRef<uint8_t> block_contents =
      GetBlockContentsFromAddress(block_address, block_size + 15);

  ProcessAllEntriesFromBlockContents(block_address, entry_processor,
                                     block_contents, block_size);
}

llvm::ArrayRef<uint8_t>
BinaryApplicationToBbDisassembler::GetBlockContentsFromAddress(
    uint64_t block_address, uint32_t block_size) const {
  auto bb_section = absl::c_lower_bound(
      address_to_section_, block_address,
      [](const std::pair<uint64_t, llvm::object::SectionRef> section_info,
         uint64_t function_address) {
        return std::get<0>(section_info) <= function_address;
      });

  QCHECK(bb_section != address_to_section_.begin());
  bb_section--;

  llvm::Expected<llvm::StringRef> section_contents =
      bb_section->second.getContents();
  QCHECK(LlvmExpectedSucceeded(section_contents));

  QCHECK_GE(block_address, bb_section->first);
  size_t bb_start_index = block_address - bb_section->first;
  const uint8_t* bb_start_address = reinterpret_cast<const uint8_t*>(
      &section_contents->data()[bb_start_index]);

  size_t section_size = section_contents->size();
  size_t available_size = section_size - bb_start_index;
  size_t actual_size =
      std::min(static_cast<size_t>(block_size), available_size);
  llvm::ArrayRef<uint8_t> block_contents(bb_start_address, actual_size);
  // llvm::ArrayRef<uint8_t> block_contents(bb_start_address, block_size);

  return block_contents;
}

BinaryApplicationToBbDisassembler::BinaryApplicationToBbDisassembler(
    const std::string& target_triple, const std::string& binary_path)
    : ApplicationToBbDisassembler(target_triple), binary_path_(binary_path) {
  llvm::Expected<llvm::object::OwningBinary<llvm::object::Binary>>
      application_binary = llvm::object::createBinary(binary_path);

  QCHECK(LlvmExpectedSucceeded(application_binary));
  binary_ = std::move(*application_binary);

  llvm::object::ObjectFile* object_file =
      llvm::cast<llvm::object::ObjectFile>(binary_.getBinary());
  QCHECK(object_file);

  // Populate the address to section map.
  for (const llvm::object::SectionRef& section : object_file->sections()) {
    // Skip all non-text sections as we only need to pull instructions from
    // these sections later, which are guaranteed to be in a text section.
    if (!section.isText()) continue;

    address_to_section_.emplace(section.getAddress(), section);
  }
}

void ApplicationToBbDisassembler::LoadBasicBlocks(
    const FunctionMapping& function_name_to_id) {
  LOG(QFATAL)
      << "LoadBasicBlocks is not implemented for the current implementation.";
}

void BinaryApplicationToBbDisassembler::LoadBasicBlocks(
    const FunctionMapping& function_name_to_id) {
  // Populate the basic block to address map.
  const absl::StatusOr<
      std::unique_ptr<mlgo::latency_model::UnstrippedBinaryProcessor>>
      unstripped_binary_processor =
          mlgo::latency_model::UnstrippedBinaryProcessor::Create(binary_path_);
  QCHECK_OK(unstripped_binary_processor);

  absl::Status possible_bbaddrmap_processing_error =
      (*unstripped_binary_processor)
          ->ProcessBBAddrMap(
              [this, &function_name_to_id](
                  const mlgo::latency_model::UnstrippedBinaryProcessor::
                      FunctionBBInfo& function_bb_info) {
                // TODO: For now, assert if we find .cold functions
                // as we need to ensure that we can handle them.
                QCHECK(!function_bb_info.function_name.ends_with(".cold"));

                const auto function_id_it =
                    function_name_to_id.function_ids().find(
                        function_bb_info.function_name);
                // Skip the function if we cannot find the name to function id
                // mapping as it should imply that no basic blocks from this
                // function are included in the traces.
                if (function_id_it ==
                    function_name_to_id.function_ids().end()) {
                  VLOG(1) << "Failed to find a function ID for "
                          << function_bb_info.function_name << "\n";
                  return;
                }

                for (uint32_t i = 0; i < function_bb_info.bb_infos.size();
                     ++i) {
                  uint32_t current_entry = 0;

                  if (function_bb_info.bb_infos[i].address == 0) {
                    continue;
                  }

                  ProcessAllEntriesInBlock(
                      function_bb_info.bb_infos[i].address,
                      function_bb_info.bb_infos[i].size,
                      [&current_entry, this, &function_id_it, i](
                          uint64_t address_offset,
                          std::vector<InstructionInfo> entry_instructions,
                          llvm::ArrayRef<uint8_t> entry_contents) -> void {
                        MachineBbId function_address_and_bb_id;
                        function_address_and_bb_id.set_function_id(
                            function_id_it->second);
                        function_address_and_bb_id.set_basic_block_id(i);
                        function_address_and_bb_id.set_entry_id(current_entry);

                        ++current_entry;

                        disassembled_instructions_.emplace(
                            function_address_and_bb_id,
                            std::move(entry_instructions));
                      });
                }
              });
  QCHECK_OK(possible_bbaddrmap_processing_error);
}

CorpusApplicationToBbDisassembler::CorpusApplicationToBbDisassembler(
    const std::string& target_triple,
    const std::vector<std::string>& module_paths, bool store_block_contents)
    : ApplicationToBbDisassembler(target_triple),
      module_paths_(module_paths),
      store_block_contents_(store_block_contents) {}

void CorpusApplicationToBbDisassembler::LoadBasicBlocks(
    const FunctionMapping& function_name_to_id) {
  // Load all of the object files.
  size_t file_index = 0;
  for (const std::string& module_path : module_paths_) {
    llvm::Expected<llvm::object::OwningBinary<llvm::object::Binary>>
        module_binary = llvm::object::createBinary(module_path);
    QCHECK(LlvmExpectedSucceeded(module_binary));

    llvm::object::ObjectFile* object_file =
        llvm::cast<llvm::object::ObjectFile>(module_binary->getBinary());
    QCHECK_NE(object_file, nullptr);

    // Populate an address to name map to use later for associating addresses
    // with the appropriate function.
    absl::flat_hash_map<std::pair<uint64_t, uint64_t>,
                        llvm::SmallSet<llvm::StringRef, 1>>
        section_index_and_offset_to_name;

    for (const llvm::object::SymbolRef& symbol : object_file->symbols()) {
      llvm::Expected<llvm::object::SymbolRef::Type> symbol_type =
          symbol.getType();
      QCHECK(LlvmExpectedSucceeded(symbol_type));
      if (*symbol_type != llvm::object::SymbolRef::ST_Function) continue;

      llvm::Expected<uint64_t> offset = symbol.getAddress();
      QCHECK(LlvmExpectedSucceeded(offset));

      llvm::Expected<llvm::object::section_iterator> symbol_section =
          symbol.getSection();
      QCHECK(LlvmExpectedSucceeded(symbol_section));
      uint64_t section_index = symbol_section.get()->getIndex();

      llvm::Expected<llvm::StringRef> symbol_name = symbol.getName();
      QCHECK(LlvmExpectedSucceeded(symbol_name));

      section_index_and_offset_to_name[std::make_pair(section_index, *offset)]
          .insert(*symbol_name);
    }

    for (const llvm::object::SectionRef& section : object_file->sections()) {
      if (!section.isText()) continue;

      // Get the basic block address map.
      const auto* elf_object_file =
          llvm::dyn_cast<llvm::object::ELFObjectFileBase>(object_file);
      QCHECK(elf_object_file);
      llvm::Expected<std::vector<llvm::object::BBAddrMap>> bb_addr_maps =
          elf_object_file->readBBAddrMap(section.getIndex());
      QCHECK(LlvmExpectedSucceeded(bb_addr_maps));

      for (const auto& bb_addr_map : *bb_addr_maps) {
        for (const llvm::object::BBAddrMap::BBEntry& entry :
             bb_addr_map.getBBEntries()) {
          uint64_t bb_offset = entry.Offset + bb_addr_map.getFunctionAddress();

          auto name_iter = section_index_and_offset_to_name.find(std::make_pair(
              section.getIndex(), bb_addr_map.getFunctionAddress()));
          if (name_iter == section_index_and_offset_to_name.end()) {
            LOG(WARNING) << "Found a BB map entry without a symbol name.";
            continue;
          }

          const llvm::SmallSet<llvm::StringRef, 1>& names = name_iter->second;
          for (const auto& name : names) {
            // TODO: For now, assert if we find .cold functions as
            // we need to ensure that we can handle them.
            QCHECK(!name.ends_with(".cold"));

            const auto function_id_it =
                function_name_to_id.function_ids().find(name);
            // Skip the function if we cannot find the name to function id
            // mapping as it should imply that no basic blocks from this
            // function are included in the traces.
            if (function_id_it == function_name_to_id.function_ids().end()) {
              VLOG(1) << "Failed to find a function ID for " << name.str()
                      << "\n";
              continue;
            }

            uint32_t current_entry = 0;

            ProcessAllEntriesInBlock(
                bb_offset, entry.Size, section,
                [&current_entry, this, &function_id_it, &entry](
                    uint64_t address_offset,
                    std::vector<InstructionInfo> partial_block_instructions,
                    llvm::ArrayRef<uint8_t> partial_block_contents) {
                  MachineBbId function_name_and_bb_id;
                  function_name_and_bb_id.set_function_id(
                      function_id_it->second);
                  function_name_and_bb_id.set_basic_block_id(entry.ID);
                  function_name_and_bb_id.set_entry_id(current_entry);

                  ++current_entry;

                  disassembled_instructions_.emplace(
                      function_name_and_bb_id,
                      std::move(partial_block_instructions));

                  if (store_block_contents_) {
                    entry_contents_.emplace(function_name_and_bb_id,
                                            partial_block_contents);
                  }
                });
          }
        }
      }
    }

    LOG(INFO) << "Finished loading " << module_path << " - " << file_index;

    ++file_index;
  }
}

void CorpusApplicationToBbDisassembler::ProcessAllEntriesInBlock(
    uint64_t bb_offset, uint32_t bb_size, llvm::object::SectionRef bb_section,
    absl::FunctionRef<void(uint64_t, std::vector<InstructionInfo>,
                           llvm::ArrayRef<uint8_t>)>
        entry_processor) const {
  llvm::ArrayRef<uint8_t> block_contents =
      GetBlockContentsFromSectionOffset(bb_offset, bb_size + 15, bb_section);

  ProcessAllEntriesFromBlockContents(bb_offset, entry_processor, block_contents,
                                     bb_size);
}

llvm::ArrayRef<uint8_t>
CorpusApplicationToBbDisassembler::GetBlockContentsFromSectionOffset(
    uint64_t bb_offset, uint32_t bb_size,
    llvm::object::SectionRef bb_section) const {
  llvm::Expected<llvm::StringRef> section_contents = bb_section.getContents();
  QCHECK(LlvmExpectedSucceeded(section_contents));

  size_t bb_start_index = bb_offset;
  const uint8_t* bb_start_offset = reinterpret_cast<const uint8_t*>(
      &section_contents->data()[bb_start_index]);

  // llvm::ArrayRef<uint8_t> block_contents(bb_start_offset, bb_size);
  size_t section_size = section_contents->size();
  size_t available_size = section_size - bb_start_index;
  size_t actual_size = std::min(static_cast<size_t>(bb_size), available_size);
  llvm::ArrayRef<uint8_t> block_contents(bb_start_offset, actual_size);

  return block_contents;
}

llvm::ArrayRef<uint8_t> CorpusApplicationToBbDisassembler::GetEntryContents(
    MachineBbId function_basic_block_id) {
  absl::ReaderMutexLock disassembling_lock(disassembling_instructions_mutex_);

  const auto basic_block_contents =
      entry_contents_.find(function_basic_block_id);

  if (basic_block_contents == entry_contents_.end() &&
      function_basic_block_id.entry_id() > 0) {
    return {};
  }

  QCHECK(basic_block_contents != entry_contents_.end())
      << function_basic_block_id.has_function_id() << ":"
      << function_basic_block_id.function_id() << ":"
      << function_basic_block_id.basic_block_id() << ":"
      << function_basic_block_id.entry_id() << "\n";

  return basic_block_contents->second;
}

}  // namespace latency_model
}  // namespace mlgo
