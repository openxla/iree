// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx940 --iree-codegen-llvmgpu-use-tile-and-fuse \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" --mlir-print-local-scope %s | FileCheck %s

// TODO: This test is still using the legacy LLVMGPU kernel config. This needs
// to be migrated to the rocdl heuristics, but for now is just physically
// located here.

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
module {
  func.func @expanded_matmul_transpose_b() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x64x2048xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<10x64x2048xf16>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x10x64x64xf16>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 64, 2048], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x64x2048xf16>> -> tensor<2x64x2048xf16>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [10, 64, 2048], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<10x64x2048xf16>> -> tensor<10x64x2048xf16>
    %5 = tensor.empty() : tensor<2x10x64x64xf16>
    %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2x10x64x64xf16>) -> tensor<2x10x64x64xf16>
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<2x64x2048xf16>, tensor<10x64x2048xf16>) outs(%6 : tensor<2x10x64x64xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %8 = arith.mulf %in, %in_0 : f16
      %9 = arith.addf %8, %out : f16
      linalg.yield %9 : f16
    } -> tensor<2x10x64x64xf16>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 10, 64, 64], strides = [1, 1, 1, 1] : tensor<2x10x64x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x10x64x64xf16>>
    return
  }
}

// CHECK-LABEL: func.func @expanded_matmul_transpose_b()
//  CHECK-SAME:   #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [16, 4, 1] subgroup_size = 64>
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     reduction = [0 : index, 0 : index, 0 : index, 0 : index, 4 : index]
//  CHECK-SAME:     thread = [1 : index, 1 : index, 1 : index, 4 : index, 0 : index]
//  CHECK-SAME:     workgroup = [1 : index, 1 : index, 4 : index, 64 : index, 0 : index]

// -----

module {
  func.func @conv_nhwc() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x258x514x768xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x3x768x256xf16>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x256x512x256xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 258, 514, 768], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x258x514x768xf16>> -> tensor<2x258x514x768xf16>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 768, 256], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x768x256xf16>> -> tensor<3x3x768x256xf16>
    %5 = tensor.empty() : tensor<2x256x512x256xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x256x512x256xf32>) -> tensor<2x256x512x256xf32>
    %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<2x258x514x768xf16>, tensor<3x3x768x256xf16>) outs(%6 : tensor<2x256x512x256xf32>) -> tensor<2x256x512x256xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 256, 512, 256], strides = [1, 1, 1, 1] : tensor<2x256x512x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x256x512x256xf32>>
    return
  }
}

// CHECK-LABEL: func.func @conv_nhwc()
//  CHECK-SAME:   #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK:   linalg.conv_2d_nhwc_hwcf {{.*}} lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     reduction = [0 : index, 0 : index, 0 : index, 0 : index, 1 : index, 3 : index, 4 : index]
//  CHECK-SAME:     thread = [1 : index, 1 : index, 1 : index, 1 : index, 0 : index, 0 : index, 0 : index]
//  CHECK-SAME:     workgroup = [1 : index, 1 : index, 1 : index, 64 : index, 0 : index, 0 : index, 0 : index]

// -----

module {
  func.func @matmul_dynamic_dim() {
          %c0 = arith.constant 0 : index
          %c32_i64 = arith.constant 32 : i64
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = arith.extui %0 : i32 to i64
          %3 = arith.extui %1 : i32 to i64
          %4 = arith.shli %3, %c32_i64 : i64
          %5 = arith.ori %2, %4 : i64
          %6 = arith.index_castui %5 : i64 to index
          %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xf16>>
          %8 = flow.dispatch.workload.ordinal %6, 0 : index
          %9 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x256xf16>>{%8}
          %10 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x256xf32>>{%8}
          %11 = flow.dispatch.tensor.load %9, offsets = [0, 0], sizes = [%8, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x256xf16>>{%8} -> tensor<?x256xf16>
          %12 = flow.dispatch.tensor.load %7, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
          %13 = tensor.empty(%8) : tensor<?x256xf32>
          %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x256xf32>) -> tensor<?x256xf32>
          %15 = linalg.matmul ins(%11, %12 : tensor<?x256xf16>, tensor<256x256xf16>) outs(%14 : tensor<?x256xf32>) -> tensor<?x256xf32>
          flow.dispatch.tensor.store %15, %10, offsets = [0, 0], sizes = [%8, 256], strides = [1, 1] : tensor<?x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x256xf32>>{%8}
          return
  }
}

// CHECK-LABEL: func.func @matmul_dynamic_dim()
//  CHECK-SAME:   #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     reduction = [0 : index, 0 : index, 4 : index]
//  CHECK-SAME:     thread = [1 : index, 1 : index, 0 : index]
//  CHECK-SAME:     workgroup = [1 : index, 64 : index, 0 : index]
