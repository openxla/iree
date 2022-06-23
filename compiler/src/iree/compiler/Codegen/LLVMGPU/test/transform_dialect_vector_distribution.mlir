// RUN: iree-opt %s --pass-pipeline='hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target-pass))' --iree-codegen-llvmgpu-use-transform-dialect=%p/transform_dialect_codegen_vector_distribution_spec.mlir | FileCheck %s

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>, #hal.descriptor_set.binding<1, storage_buffer>]>]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}>

hal.executable private @reduce_dispatch_0 {
  hal.executable.variant public @cuda_nvptx_fb, target = #executable_target_cuda_nvptx_fb {
    hal.executable.export public @reduce_dispatch_0 ordinal(0) layout(#executable_layout) { workgroup_size = [64: index, 1: index, 1: index] }
    builtin.module {
      func.func @reduce_dispatch_0() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %0 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : memref<128xf32>
        memref.assume_alignment %0, 64 : memref<128xf32>
        %1 = gpu.thread_id  x
        %2 = arith.cmpi ult, %1, %c1 : index

        // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
        // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
        // CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
        // CHECK-DAG: %[[V128:.*]] = arith.constant dense<1.000000e+00> : vector<128xf32>
        // CHECK: %[[TIDX:.*]] = gpu.thread_id  x
        // CHECK: %[[COND32:.*]] = arith.cmpi ult, %[[TIDX]], %[[C32]] : index
        // Single-warp guard filters out threads 32-63.
        // CHECK: scf.if %[[COND32]] {
        // CHECK:   %[[COND1:.*]] = arith.cmpi eq, %[[TIDX]], %[[C0]] : index
        // CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<128xf32, 3>
        // Single-thread guard runs on thread 0 only.
        // CHECK:   scf.if %[[COND1]] {
        // CHECK:     vector.store %{{.*}} : memref<128xf32, 3>, vector<128xf32>
        // CHECK:   %[[IDX:.*]] = arith.muli %[[TIDX]], %[[C4]] : index
        // CHECK:   %[[LOADED:.*]] = vector.load %{{.*}}[%[[IDX]]] : memref<128xf32, 3>, vector<4xf32>
        // CHECK:   vector.transfer_write %[[LOADED]], %{{.*}} {in_bounds = [true]} : vector<4xf32>, memref<128xf32>
        scf.if %2 {
          %3 = arith.constant dense<1.0> : vector<128xf32>
          vector.transfer_write %3, %0[%c0] : vector<128xf32>, memref<128xf32>
        }

        
        return
      }
    }
  }
}
