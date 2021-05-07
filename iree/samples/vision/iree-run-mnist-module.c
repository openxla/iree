// Copyright 2021 Google LLC
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

// This sample uses iree/tools/utils/image_util to load an hand-writing image
// to the iree hall buffer view, and runs on the bytecode module built from
// mnist.mlir with dylib-llvm-aot backend. Other vision application can follow
// the similar flow.

#include <float.h>

#include "iree/runtime/api.h"
#include "iree/samples/vision/mnist_bytecode_module_c.h"
#include "iree/tools/utils/image_util.h"

iree_status_t Run() {
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(IREE_API_VERSION_LATEST,
                                           &instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &instance));

  // TODO(#5724): move device selection into the compiled modules.
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view("dylib"), &device));

  // Create one session per loaded module to hold the module state.
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session));
  iree_hal_device_release(device);

  const struct iree_file_toc_t* module_file = mnist_bytecode_module_c_create();

  IREE_RETURN_IF_ERROR(iree_runtime_session_append_bytecode_module_from_memory(
      session, iree_make_const_byte_span(module_file->data, module_file->size),
      iree_allocator_null()));

  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.predict"), &call));

  // Prepare the input hal buffer view with image_util library.
  iree_hal_buffer_view_t* buffer_view = NULL;
  const char kInputImage[] = "mnist_test.png";
  iree_hal_dim_t buffer_shape[] = {1, 28, 28, 1};
  iree_hal_element_type_t hal_element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
  float input_range[2] = {0.0f, 1.0f};
  IREE_RETURN_IF_ERROR(iree_tools_utils_buffer_view_from_image_rescaled(
                           iree_make_cstring_view(kInputImage), buffer_shape,
                           IREE_ARRAYSIZE(buffer_shape), hal_element_type,
                           iree_hal_device_allocator(device), input_range,
                           IREE_ARRAYSIZE(input_range), &buffer_view),
                       "load image");
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_inputs_push_back_buffer_view(&call, buffer_view));
  iree_hal_buffer_view_release(buffer_view);

  IREE_RETURN_IF_ERROR(iree_runtime_call_invoke(&call, /*flags=*/0));

  // Get the result buffers from the invocation.
  iree_hal_buffer_view_t* ret_buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_outputs_pop_front_buffer_view(&call, &ret_buffer_view));

  // Read back the results and ensure we have the right values.
  iree_hal_buffer_mapping_t mapped_memory;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(ret_buffer_view), IREE_HAL_MEMORY_ACCESS_READ,
      0, IREE_WHOLE_BUFFER, &mapped_memory));
  float result_val = FLT_MIN;
  int result_idx = 0;
  const float* data_ptr = (const float*)mapped_memory.contents.data;
  for (int i = 0; i < mapped_memory.contents.data_length / sizeof(float); ++i) {
    if (data_ptr[i] > result_val) {
      result_val = data_ptr[i];
      result_idx = i;
    }
  }
  iree_hal_buffer_unmap_range(&mapped_memory);
  // Get the highest index from the output.
  if (result_idx != 4) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "detection result error. expect 4, get %d",
                            result_idx);
  }
  fprintf(stdout, "Detected number: %d\n", result_idx);
  iree_hal_buffer_view_release(ret_buffer_view);

  iree_runtime_call_deinitialize(&call);
  iree_runtime_session_release(session);
  iree_runtime_instance_release(instance);
  return iree_ok_status();
}

int main() {
  iree_status_t result = Run();
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_ignore(result);
    return -1;
  }
  iree_status_ignore(result);
  return 0;
}
