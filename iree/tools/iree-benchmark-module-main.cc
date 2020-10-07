// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "absl/flags/flag.h"
#include "absl/flags/internal/parse.h"
#include "absl/flags/usage.h"
#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"
#include "iree/base/file_io.h"
#include "iree/base/init.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/tools/utils/vm_util.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"

// TODO(gcmn): Allow stdin in a non-gross way. The benchmark framework invokes
// the benchmarking function multiple times, so we have to do something to only
// process stdin once. Probably requires dynamic benchmark registration.
ABSL_FLAG(std::string, module_file, "",
          "File containing the module to load that contains the entry "
          "function. Required and cannot be stdin.");

ABSL_FLAG(std::string, entry_function, "",
          "Name of a function contained in the module specified by module_file "
          "to run.");

ABSL_FLAG(std::string, driver, "vmla", "Backend driver to use.");

ABSL_FLAG(std::vector<std::string>, function_inputs, {},
          "A comma-separated list of of input buffers of the format:"
          "[shape]xtype=[value]\n"
          "2x2xi32=1 2 3 4\n"
          "Optionally, brackets may be used to separate the element values. "
          "They are ignored by the parser.\n"
          "2x2xi32=[[1 2][3 4]]\n"
          "Due to the absence of repeated flags in absl, commas should not be "
          "used to separate elements. They are reserved for separating input "
          "values:\n"
          "2x2xi32=[[1 2][3 4]], 1x2xf32=[[1 2]]");

ABSL_FLAG(std::string, function_inputs_file, "",
          "Provides a file for input shapes and optional values (see "
          "ParseToVariantListFromFile in vm_util.h for details)");

namespace iree {
namespace {

Status GetModuleContentsFromFlags(std::string& module_data) {
  auto module_file = absl::GetFlag(FLAGS_module_file);
  IREE_ASSIGN_OR_RETURN(module_data, file_io::GetFileContents(module_file));
  return iree::OkStatus();
}

// Creates VM function instance with its inputs.
Status PrepareIREEVMFunction(
    iree_vm_instance_t** instance, iree_hal_device_t** device,
    iree_vm_module_t** hal_module, iree_vm_context_t** context,
    iree_vm_module_t** input_module, iree_vm_function_t& function,
    iree::vm::ref<iree_vm_list_t>& inputs,
    std::vector<iree::RawSignatureParser::Description>& output_descs,
    const std::string& function_name, const std::string& module_data) {
  IREE_RETURN_IF_ERROR(iree_hal_module_register_types());
  IREE_RETURN_IF_ERROR(
      iree_vm_instance_create(iree_allocator_system(), instance));

  // Create IREE's device and module.
  IREE_RETURN_IF_ERROR(
      iree::CreateDevice(absl::GetFlag(FLAGS_driver), device));
  IREE_RETURN_IF_ERROR(CreateHalModule(*device, hal_module));
  IREE_RETURN_IF_ERROR(LoadBytecodeModule(module_data, input_module));

  // Order matters. The input module will likely be dependent on the hal
  // module.
  std::array<iree_vm_module_t*, 2> modules = {*hal_module, *input_module};
  IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
      *instance, modules.data(), modules.size(), iree_allocator_system(),
      context));

  IREE_RETURN_IF_ERROR((*input_module)->lookup_function(
      (*input_module)->self, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      iree_string_view_t{function_name.data(), function_name.size()},
      &function));
  IREE_RETURN_IF_ERROR(ValidateFunctionAbi(function));

  // Construct inputs.
  IREE_ASSIGN_OR_RETURN(auto input_descs, ParseInputSignature(function));

  if (!absl::GetFlag(FLAGS_function_inputs_file).empty()) {
    IREE_ASSIGN_OR_RETURN(inputs,
                          ParseToVariantListFromFile(
                              input_descs, iree_hal_device_allocator(*device),
                              absl::GetFlag(FLAGS_function_inputs_file)));
  } else {
    IREE_ASSIGN_OR_RETURN(
        inputs,
        ParseToVariantList(input_descs, iree_hal_device_allocator(*device),
                           absl::GetFlag(FLAGS_function_inputs)));
  }

  // Creates output singnature.
  IREE_ASSIGN_OR_RETURN(output_descs, ParseOutputSignature(function));
  return iree::OkStatus();
}

void RegisterModuleBenchmarks(
    Status module_data_status, Status prepare_vm_func_status,
    std::string& function_name, iree_vm_context_t* context,
    iree_vm_function_t function, iree_vm_list_t* inputs,
    const std::vector<RawSignatureParser::Description>& output_descs) {
  auto benchmark_name = "BM_" + function_name;
  benchmark::RegisterBenchmark(
      benchmark_name.c_str(),
      [module_data_status, prepare_vm_func_status, context, function, inputs,
       output_descs](benchmark::State& state) -> void {
        IREE_CHECK_OK(module_data_status);
        IREE_CHECK_OK(prepare_vm_func_status);
        // Warmup run step.
        {
          vm::ref<iree_vm_list_t> outputs;
          IREE_CHECK_OK(iree_vm_list_create(/*element_type=*/nullptr,
                                            output_descs.size(),
                                            iree_allocator_system(), &outputs));
          IREE_CHECK_OK(iree_vm_invoke(context, function, /*policy=*/nullptr,
                                       inputs, outputs.get(),
                                       iree_allocator_system()));
        }
        // Benchmarking loop.
        for (auto _ : state) {
          // No status conversions and conditional returns in the benchmarked
          // inner loop.
          vm::ref<iree_vm_list_t> outputs;
          IREE_CHECK_OK(iree_vm_list_create(/*element_type=*/nullptr,
                                            output_descs.size(),
                                            iree_allocator_system(), &outputs));
          IREE_CHECK_OK(iree_vm_invoke(context, function, /*policy=*/nullptr,
                                       inputs, outputs.get(),
                                       iree_allocator_system()));
        }
      })
      // By default only the main thread is included in CPU time. Include all
      // the threads instead.
      ->MeasureProcessCPUTime()
      // To make single and multi-threaded benchmarks more comparable, use the
      // wall time to determine how many iterations to run. See
      // https://github.com/google/benchmark#cpu-timers,
      ->UseRealTime()
      // Report timing in milliseconds, which is the general order of magnitude
      // of model runs. The benchmark framework will print with precision
      // between 0 and 3 places after the decimal while aiming for three
      // significant digits. If we end up wanting precision beyond microseconds,
      // we can make this setting configurable with a custom command line flag.
      ->Unit(benchmark::kMillisecond);
}
}  // namespace
}  // namespace iree

int main(int argc, char** argv) {
  // We have to contend with two flag parsing libraries here: absl's and
  // benchmark's. To make matters worse, both define the `--help` flag. To
  // ensure that each is able to parse its own flags, we use an absl "internal"
  // function (still with public visibility) to parse while ignoring undefined
  // flags. If it sees `--help` it will exit here, so we include the benchmark
  // library usage information in the manually-set help output. Then we let
  // benchmark parse its flags. Finally we call the normal initialization
  // function to do other IREE initialization including flag parsing with
  // normal options. Any remaining flags will be unknown and result in an error.
  absl::SetProgramUsageMessage(
      "iree-benchmark-module \n"
      "    --module_file=module.vmfb\n"
      "    --entry_function=exported_function_to_benchmark\n"
      "    [--function_inputs=2xi32=1 2,1x2xf32=2 1 | "
      "     --function_inputs_file=file_with_function_inputs]\n"
      "    [--driver=vmla]\n"
      "\n\n"
      "  Optional flags from third_party/benchmark/src/benchmark.cc:\n"
      "    [--benchmark_list_tests={true|false}]\n"
      "    [--benchmark_filter=<regex>]\n"
      "    [--benchmark_min_time=<min_time>]\n"
      "    [--benchmark_repetitions=<num_repetitions>]\n"
      "    [--benchmark_report_aggregates_only={true|false}]\n"
      "    [--benchmark_display_aggregates_only={true|false}]\n"
      "    [--benchmark_format=<console|json|csv>]\n"
      "    [--benchmark_out=<filename>]\n"
      "    [--benchmark_out_format=<json|console|csv>]\n"
      "    [--benchmark_color={auto|true|false}]\n"
      "    [--benchmark_counters_tabular={true|false}]\n"
      "    [--v=<verbosity>]\n");
  absl::flags_internal::ParseCommandLineImpl(
      argc, argv, absl::flags_internal::ArgvListAction::kRemoveParsedArgs,
      absl::flags_internal::UsageFlagsAction::kHandleUsage,
      absl::flags_internal::OnUndefinedFlag::kIgnoreUndefined);
  ::benchmark::Initialize(&argc, argv);
  iree::InitializeEnvironment(&argc, &argv);

  iree_vm_instance_t* instance = nullptr;
  iree_hal_device_t* device = nullptr;
  iree_vm_module_t* hal_module = nullptr;
  iree_vm_context_t* context = nullptr;
  iree_vm_module_t* input_module = nullptr;
  iree_vm_function_t function;
  iree::vm::ref<iree_vm_list_t> inputs;
  std::vector<iree::RawSignatureParser::Description> output_descs;
  std::string module_data;
  auto function_name = absl::GetFlag(FLAGS_entry_function);

  // Capture status checking will be delayed at benchmarking runtime.
  auto module_data_status = iree::GetModuleContentsFromFlags(module_data);
  auto prepare_vm_func_status = iree::PrepareIREEVMFunction(
      &instance, &device, &hal_module, &context, &input_module, function,
      inputs, output_descs, function_name, module_data);
  if (prepare_vm_func_status.ok()) {
    assert(instance);
    assert(device);
    assert(hal_module);
    assert(context);
    assert(input_module);
    assert(inputs.get());
  }

  // Register function benchmarks...
  iree::RegisterModuleBenchmarks(module_data_status, prepare_vm_func_status,
                                 function_name, context, function, inputs.get(),
                                 output_descs);

  // Run benchmarks...
  ::benchmark::RunSpecifiedBenchmarks();

  // Cleanup...
  inputs.reset();
  iree_vm_module_release(hal_module);
  iree_vm_module_release(input_module);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);
  return 0;
}
