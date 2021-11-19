// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/cuda/graph_command_buffer.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/inline_array.h"
#include "iree/base/tracing.h"
#include "iree/hal/cuda/cuda_buffer.h"
#include "iree/hal/cuda/cuda_event.h"
#include "iree/hal/cuda/dynamic_symbols.h"
#include "iree/hal/cuda/executable_layout.h"
#include "iree/hal/cuda/native_executable.h"
#include "iree/hal/cuda/status_util.h"

#define CUDA_MAX_LEAF_NODES 16

// Command buffer implementation that directly maps to cuda graph.
// This records the commands on the calling thread without additional threading
// indirection.
typedef struct iree_hal_cuda_graph_command_buffer_t {
  iree_hal_resource_t resource;
  iree_hal_cuda_context_wrapper_t* context;
  iree_hal_command_buffer_mode_t mode;
  iree_hal_command_category_t allowed_categories;
  iree_hal_queue_affinity_t queue_affinity;
  CUgraph graph;
  CUgraphExec exec;
  // Keep track of an array of leaf nodes in the graph. Whenever an event is
  // raised all the leaf nodes will be parents of the event node.
  CUgraphNode leaf_nodes[CUDA_MAX_LEAF_NODES];
  // Number of valid nodes in leaf_nodes.
  size_t num_leaf_nodes;
  // Keep track of the node any new dispatch needs to depend on. For simplicity
  // we make all the dependency merged into a single node. This may require
  // adding empty nodes. As an optimization we could try to reduce the number of
  // empty nodes in the future.
  CUgraphNode last_sync_node;
  // Keep track of the current set of kernel arguments.
  void* current_descriptor[];
} iree_hal_cuda_graph_command_buffer_t;

#define IREE_HAL_CUDA_MAX_BINDING_COUNT 64
// Kernel arguments contains binding and push constants.
#define IREE_HAL_CUDA_MAX_KERNEL_ARG 128

extern const iree_hal_command_buffer_vtable_t
    iree_hal_cuda_graph_command_buffer_vtable;

static iree_hal_cuda_graph_command_buffer_t*
iree_hal_cuda_graph_command_buffer_cast(iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_graph_command_buffer_vtable);
  return (iree_hal_cuda_graph_command_buffer_t*)base_value;
}

iree_status_t iree_hal_cuda_graph_command_buffer_create(
    iree_hal_cuda_context_wrapper_t* context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  CUgraph graph = NULL;
  CUDA_RETURN_IF_ERROR(context->syms, cuGraphCreate(&graph, /*flags=*/0),
                       "cuGraphCreate");
  iree_hal_cuda_graph_command_buffer_t* command_buffer = NULL;
  size_t total_size = sizeof(*command_buffer) +
                      IREE_HAL_CUDA_MAX_KERNEL_ARG * sizeof(void*) +
                      IREE_HAL_CUDA_MAX_KERNEL_ARG * sizeof(CUdeviceptr);
  iree_status_t status = iree_allocator_malloc(
      context->host_allocator, total_size, (void**)&command_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_cuda_graph_command_buffer_vtable,
                                 &command_buffer->resource);
    command_buffer->context = context;
    command_buffer->mode = mode;
    command_buffer->allowed_categories = command_categories;
    command_buffer->queue_affinity = queue_affinity;
    command_buffer->graph = graph;
    command_buffer->exec = NULL;
    command_buffer->num_leaf_nodes = 0;
    command_buffer->last_sync_node = NULL;

    CUdeviceptr* device_ptrs =
        (CUdeviceptr*)(command_buffer->current_descriptor +
                       IREE_HAL_CUDA_MAX_KERNEL_ARG);
    for (size_t i = 0; i < IREE_HAL_CUDA_MAX_KERNEL_ARG; i++) {
      command_buffer->current_descriptor[i] = &device_ptrs[i];
    }

    *out_command_buffer = (iree_hal_command_buffer_t*)command_buffer;
  } else {
    context->syms->cuGraphDestroy(graph);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda_graph_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (command_buffer->graph != NULL) {
    CUDA_IGNORE_ERROR(command_buffer->context->syms,
                      cuGraphDestroy(command_buffer->graph));
  }
  if (command_buffer->exec != NULL) {
    CUDA_IGNORE_ERROR(command_buffer->context->syms,
                      cuGraphExecDestroy(command_buffer->exec));
  }
  iree_allocator_free(command_buffer->context->host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

CUgraphExec iree_hal_cuda_graph_command_buffer_handle(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);
  return command_buffer->exec;
}

bool iree_hal_cuda_graph_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_command_buffer_dyn_cast(
      command_buffer, &iree_hal_cuda_graph_command_buffer_vtable);
}

static void* iree_hal_cuda_graph_command_buffer_dyn_cast(
    iree_hal_command_buffer_t* command_buffer, const void* vtable) {
  if (vtable == &iree_hal_cuda_graph_command_buffer_vtable) {
    IREE_HAL_ASSERT_TYPE(command_buffer, vtable);
    return command_buffer;
  }
  return NULL;
}

static iree_hal_command_buffer_mode_t iree_hal_cuda_graph_command_buffer_mode(
    const iree_hal_command_buffer_t* base_command_buffer) {
  const iree_hal_cuda_graph_command_buffer_t* command_buffer =
      (const iree_hal_cuda_graph_command_buffer_t*)(base_command_buffer);
  return command_buffer->mode;
}

static iree_hal_command_category_t
iree_hal_cuda_graph_command_buffer_allowed_categories(
    const iree_hal_command_buffer_t* base_command_buffer) {
  const iree_hal_cuda_graph_command_buffer_t* command_buffer =
      (const iree_hal_cuda_graph_command_buffer_t*)(base_command_buffer);
  return command_buffer->allowed_categories;
}

static iree_status_t iree_hal_cuda_graph_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  // Nothing to do.
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_graph_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);

  size_t num_nodes;
  CUDA_RETURN_IF_ERROR(command_buffer->context->syms,
                       cuGraphGetNodes(command_buffer->graph, NULL, &num_nodes),
                       "cuGraphGetNodes");

  CUgraphNode error_node;
  iree_status_t status =
      CU_RESULT_TO_STATUS(command_buffer->context->syms,
                          cuGraphInstantiate(&command_buffer->exec,
                                             command_buffer->graph, &error_node,
                                             /*logBuffer=*/NULL,
                                             /* bufferSize=*/0));
  if (iree_status_is_ok(status)) {
    CUDA_IGNORE_ERROR(command_buffer->context->syms,
                      cuGraphDestroy(command_buffer->graph));
  }
  command_buffer->graph = NULL;
  return iree_ok_status();
}

static void iree_hal_cuda_graph_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  // TODO(benvanik): tracy event stack.
}

static void iree_hal_cuda_graph_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  // TODO(benvanik): tracy event stack.
}

static iree_status_t iree_hal_merge_node_edges(
    iree_hal_cuda_graph_command_buffer_t* command_buffer, size_t num_nodes,
    const CUgraphNode* nodes, CUgraphNode* merged_node) {
  if (num_nodes == 0) {
    *merged_node = NULL;
  } else if (num_nodes == 1) {
    *merged_node = nodes[0];
  } else if (num_nodes > 1) {
    CUDA_RETURN_IF_ERROR(command_buffer->context->syms,
                         cuGraphAddEmptyNode(merged_node, command_buffer->graph,
                                             nodes, num_nodes),
                         "cuGraphAddEmptyNode");
  }
  return iree_ok_status();
}

static iree_status_t iree_cuda_flush_dep_nodes(
    iree_hal_cuda_graph_command_buffer_t* command_buffer) {
  if (command_buffer->num_leaf_nodes <= 1) {
    return iree_ok_status();
  }
  CUgraphNode dummy_node;
  // If there are more nodes than our queue size add a dummy node with all the
  // edges.
  CUDA_RETURN_IF_ERROR(command_buffer->context->syms,
                       cuGraphAddEmptyNode(&dummy_node, command_buffer->graph,
                                           command_buffer->leaf_nodes,
                                           command_buffer->num_leaf_nodes),
                       "cuGraphAddEmptyNode");
  command_buffer->num_leaf_nodes = 0;
  command_buffer->leaf_nodes[command_buffer->num_leaf_nodes++] = dummy_node;
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_graph_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);
  iree_status_t status = iree_cuda_flush_dep_nodes(command_buffer);
  if (command_buffer->num_leaf_nodes > 0) {
    command_buffer->num_leaf_nodes = 0;
    command_buffer->last_sync_node = command_buffer->leaf_nodes[0];
  }
  return status;
}

static iree_status_t iree_hal_cuda_graph_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);
  iree_status_t status = iree_cuda_flush_dep_nodes(command_buffer);
  if (command_buffer->num_leaf_nodes > 0) {
    iree_hal_cuda_event_set_sync_node(event, command_buffer->leaf_nodes[0]);
  }
  return status;
}

static iree_status_t iree_hal_cuda_graph_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_cuda_event_set_sync_node(event, NULL);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_graph_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);
  iree_host_size_t edge_count =
      event_count + (command_buffer->last_sync_node ? 1 : 0);
  iree_inline_array(CUgraphNode, node_array, edge_count,
                    command_buffer->context->host_allocator);
  for (iree_host_size_t i = 0; i < event_count; i++) {
    CUgraphNode syncNode = iree_hal_cuda_event_get_sync_node(events[i]);
    *iree_inline_array_at(node_array, i) = syncNode;
  }
  if (command_buffer->last_sync_node) {
    *iree_inline_array_at(node_array, event_count) =
        command_buffer->last_sync_node;
  }
  CUgraphNode mergedNode;
  iree_hal_merge_node_edges(command_buffer, event_count,
                            iree_inline_array_data(node_array), &mergedNode);
  command_buffer->last_sync_node = mergedNode;
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_graph_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_t* buffer) {
  // nothing to do.
  return iree_ok_status();
}

// Splats a pattern value of 1, 2, or 4 bytes out to a 4 byte value.
static uint32_t iree_hal_cuda_splat_pattern(const void* pattern,
                                            size_t pattern_length) {
  switch (pattern_length) {
    case 1: {
      uint32_t pattern_value = *(const uint8_t*)(pattern);
      return (pattern_value << 24) | (pattern_value << 16) |
             (pattern_value << 8) | pattern_value;
    }
    case 2: {
      uint32_t pattern_value = *(const uint16_t*)(pattern);
      return (pattern_value << 16) | pattern_value;
    }
    case 4: {
      uint32_t pattern_value = *(const uint32_t*)(pattern);
      return pattern_value;
    }
    default:
      return 0;  // Already verified that this should not be possible.
  }
}

static size_t iree_cuda_get_dep_nodes(
    iree_hal_cuda_graph_command_buffer_t* command_buffer,
    CUgraphNode (*dep)[CUDA_MAX_LEAF_NODES]) {
  if (command_buffer->last_sync_node == NULL) return 0;
  (*dep)[0] = command_buffer->last_sync_node;
  return 1;
}

static iree_status_t iree_cuda_add_leaf_node(
    iree_hal_cuda_graph_command_buffer_t* command_buffer, CUgraphNode node) {
  iree_status_t status = iree_ok_status();
  if (command_buffer->num_leaf_nodes == CUDA_MAX_LEAF_NODES) {
    status = iree_cuda_flush_dep_nodes(command_buffer);
  }
  command_buffer->leaf_nodes[command_buffer->num_leaf_nodes++] = node;
  return status;
}

static iree_status_t iree_hal_cuda_graph_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);

  CUdeviceptr target_device_buffer = iree_hal_cuda_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_buffer));
  target_offset += iree_hal_buffer_byte_offset(target_buffer);
  uint32_t dword_pattern = iree_hal_cuda_splat_pattern(pattern, pattern_length);
  CUDA_MEMSET_NODE_PARAMS params = {
      .dst = target_device_buffer + target_offset,
      .elementSize = pattern_length,
      // width in number of elements despite what driver documentation says.
      .width = length / pattern_length,
      .height = 1,
      .value = dword_pattern,
  };
  CUgraphNode dep[CUDA_MAX_LEAF_NODES];
  size_t numNode = iree_cuda_get_dep_nodes(command_buffer, &dep);
  CUgraphNode node;
  CUDA_RETURN_IF_ERROR(
      command_buffer->context->syms,
      cuGraphAddMemsetNode(&node, command_buffer->graph, dep, numNode, &params,
                           command_buffer->context->cu_context),
      "cuGraphAddMemsetNode");

  return iree_cuda_add_leaf_node(command_buffer, node);
}

static iree_status_t iree_hal_cuda_graph_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need cuda implementation");
}

static iree_status_t iree_hal_cuda_graph_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);

  CUdeviceptr target_device_buffer = iree_hal_cuda_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_buffer));
  target_offset += iree_hal_buffer_byte_offset(target_buffer);
  CUdeviceptr source_device_buffer = iree_hal_cuda_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(source_buffer));
  source_offset += iree_hal_buffer_byte_offset(source_buffer);
  CUDA_MEMCPY3D params = {
      .Depth = 1,
      .Height = 1,
      .WidthInBytes = length,
      .dstDevice = target_device_buffer,
      .srcDevice = source_device_buffer,
      .srcXInBytes = source_offset,
      .dstXInBytes = target_offset,
      .srcMemoryType = CU_MEMORYTYPE_DEVICE,
      .dstMemoryType = CU_MEMORYTYPE_DEVICE,
  };
  CUgraphNode dep[CUDA_MAX_LEAF_NODES];
  size_t numNode = iree_cuda_get_dep_nodes(command_buffer, &dep);
  CUgraphNode node;
  CUDA_RETURN_IF_ERROR(
      command_buffer->context->syms,
      cuGraphAddMemcpyNode(&node, command_buffer->graph, dep, numNode, &params,
                           command_buffer->context->cu_context),
      "cuGraphAddMemcpyNode");
  return iree_cuda_add_leaf_node(command_buffer, node);
}

static iree_status_t iree_hal_cuda_graph_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);
  iree_host_size_t constant_base_index =
      iree_hal_cuda_push_constant_index(executable_layout) +
      offset / sizeof(int32_t);
  for (iree_host_size_t i = 0; i < values_length / sizeof(int32_t); i++) {
    *((uint32_t*)command_buffer->current_descriptor[i + constant_base_index]) =
        ((uint32_t*)values)[i];
  }
  return iree_ok_status();
}

// Tie together the binding index and its index in |bindings| array.
typedef struct {
  uint32_t index;
  uint32_t binding;
} iree_hal_cuda_binding_mapping_t;

// Helper to sort the binding based on their binding index.
static int compare_binding_index(const void* a, const void* b) {
  const iree_hal_cuda_binding_mapping_t buffer_a =
      *(const iree_hal_cuda_binding_mapping_t*)a;
  const iree_hal_cuda_binding_mapping_t buffer_b =
      *(const iree_hal_cuda_binding_mapping_t*)b;
  return buffer_a.binding < buffer_b.binding ? -1 : 1;
}

static iree_status_t iree_hal_cuda_graph_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);
  iree_host_size_t base_binding =
      iree_hal_cuda_base_binding_index(executable_layout, set);
  // Convention with the compiler side. We map bindings to kernel argument.
  // We compact the bindings to get a dense set of arguments and keep them order
  // based on the binding index.
  // Sort the binding based on the binding index and map the array index to the
  // argument index.
  iree_hal_cuda_binding_mapping_t binding_used[IREE_HAL_CUDA_MAX_BINDING_COUNT];
  for (iree_host_size_t i = 0; i < binding_count; i++) {
    iree_hal_cuda_binding_mapping_t buffer = {i, bindings[i].binding};
    binding_used[i] = buffer;
  }
  qsort(binding_used, binding_count, sizeof(iree_hal_cuda_binding_mapping_t),
        compare_binding_index);
  assert(binding_count < IREE_HAL_CUDA_MAX_BINDING_COUNT &&
         "binding count larger than the max expected.");
  for (iree_host_size_t i = 0; i < binding_count; i++) {
    iree_hal_descriptor_set_binding_t binding = bindings[binding_used[i].index];
    CUdeviceptr device_ptr =
        iree_hal_cuda_buffer_device_pointer(
            iree_hal_buffer_allocated_buffer(binding.buffer)) +
        iree_hal_buffer_byte_offset(binding.buffer) + binding.offset;
    *((CUdeviceptr*)command_buffer->current_descriptor[i + base_binding]) =
        device_ptr;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_graph_command_buffer_bind_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_hal_descriptor_set_t* descriptor_set,
    iree_host_size_t dynamic_offset_count,
    const iree_device_size_t* dynamic_offsets) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need cuda implementation");
}

static iree_status_t iree_hal_cuda_graph_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);

  int32_t block_size_x, block_size_y, block_size_z;
  IREE_RETURN_IF_ERROR(iree_hal_cuda_native_executable_block_size(
      executable, entry_point, &block_size_x, &block_size_y, &block_size_z));
  CUDA_KERNEL_NODE_PARAMS params = {
      .func = iree_hal_cuda_native_executable_for_entry_point(executable,
                                                              entry_point),
      .blockDimX = block_size_x,
      .blockDimY = block_size_y,
      .blockDimZ = block_size_z,
      .gridDimX = workgroup_x,
      .gridDimY = workgroup_y,
      .gridDimZ = workgroup_z,
      .kernelParams = command_buffer->current_descriptor,
  };
  // Serialize all the nodes for now.
  CUgraphNode dep[CUDA_MAX_LEAF_NODES];
  size_t numNodes = iree_cuda_get_dep_nodes(command_buffer, &dep);
  CUgraphNode node;
  CUDA_RETURN_IF_ERROR(command_buffer->context->syms,
                       cuGraphAddKernelNode(&node, command_buffer->graph, dep,
                                            numNodes, &params),
                       "cuGraphAddKernelNode");
  return iree_cuda_add_leaf_node(command_buffer, node);
}

static iree_status_t iree_hal_cuda_graph_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need cuda implementation");
}

CUgraphExec iree_hal_cuda_graph_command_buffer_exec(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      (iree_hal_cuda_graph_command_buffer_t*)iree_hal_command_buffer_dyn_cast(
          base_command_buffer, &iree_hal_cuda_graph_command_buffer_vtable);
  IREE_ASSERT_TRUE(command_buffer);
  return command_buffer->exec;
}

const iree_hal_command_buffer_vtable_t
    iree_hal_cuda_graph_command_buffer_vtable = {
        .destroy = iree_hal_cuda_graph_command_buffer_destroy,
        .dyn_cast = iree_hal_cuda_graph_command_buffer_dyn_cast,
        .mode = iree_hal_cuda_graph_command_buffer_mode,
        .allowed_categories =
            iree_hal_cuda_graph_command_buffer_allowed_categories,
        .begin = iree_hal_cuda_graph_command_buffer_begin,
        .end = iree_hal_cuda_graph_command_buffer_end,
        .begin_debug_group =
            iree_hal_cuda_graph_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_cuda_graph_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_cuda_graph_command_buffer_execution_barrier,
        .signal_event = iree_hal_cuda_graph_command_buffer_signal_event,
        .reset_event = iree_hal_cuda_graph_command_buffer_reset_event,
        .wait_events = iree_hal_cuda_graph_command_buffer_wait_events,
        .discard_buffer = iree_hal_cuda_graph_command_buffer_discard_buffer,
        .fill_buffer = iree_hal_cuda_graph_command_buffer_fill_buffer,
        .update_buffer = iree_hal_cuda_graph_command_buffer_update_buffer,
        .copy_buffer = iree_hal_cuda_graph_command_buffer_copy_buffer,
        .push_constants = iree_hal_cuda_graph_command_buffer_push_constants,
        .push_descriptor_set =
            iree_hal_cuda_graph_command_buffer_push_descriptor_set,
        .bind_descriptor_set =
            iree_hal_cuda_graph_command_buffer_bind_descriptor_set,
        .dispatch = iree_hal_cuda_graph_command_buffer_dispatch,
        .dispatch_indirect =
            iree_hal_cuda_graph_command_buffer_dispatch_indirect,
};
