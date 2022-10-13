// RUN: iree-opt --split-input-file --pass-pipeline='hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true}))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @batch_matmul_16x4096x40x4096 {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader], []>, AMD:DiscreteGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64>>
    }> {
    hal.executable.export @batch_matmul_16x4096x40x4096 layout(#pipeline_layout)
    builtin.module {
      func.func @batch_matmul_16x4096x40x4096() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:16x4096x4096xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:16x4096x40xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readwrite:16x4096x40xf32>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 4096, 4096], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:16x4096x4096xf32> -> tensor<16x4096x4096xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [16, 4096, 40], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:16x4096x40xf32> -> tensor<16x4096x40xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [16, 4096, 40], strides = [1, 1, 1] : !flow.dispatch.tensor<readwrite:16x4096x40xf32> -> tensor<16x4096x40xf32>
        %6 = linalg.batch_matmul ins(%3, %4 : tensor<16x4096x4096xf32>, tensor<16x4096x40xf32>) outs(%5 : tensor<16x4096x40xf32>) -> tensor<16x4096x40xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [16, 4096, 40], strides = [1, 1, 1] : tensor<16x4096x40xf32> -> !flow.dispatch.tensor<readwrite:16x4096x40xf32>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1024, 8], [1, 8, 4], [0, 0, 0, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize>
//      CHECK: hal.executable.export public @batch_matmul_16x4096x40x4096
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [2 : index, 128 : index, 1 : index]
//      CHECK: func.func @batch_matmul_16x4096x40x4096()
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

