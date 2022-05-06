// RUN: iree-dialects-opt --linalg-transform-interp %s | FileCheck %s

// This test is verifying that a non-trivial 2*tiling+padding+vectorization transformation completes successfully

// CHECK-LABEL: func @matmul_tensors(
func @matmul_tensors(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32> { linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // Pack transposed padding of 1st operand.
  //      CHECK:    tensor.pad
  //      CHECK:    linalg.generic

  // Pack padding of 2nd operand.
  //      CHECK:    tensor.pad

  //      CHECK:      scf.for
  //      CHECK:        scf.for
  //      CHECK:          scf.for
  //      CHECK:            scf.for
  //      CHECK:              scf.for
  //      CHECK:                linalg.generic
  //      CHECK:                vector.contract
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

  return %0 : tensor<128x128xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @pdl_target: benefit(1) {
    %args = operands
    %results = types
    %0 = operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    %1 = pdl.attribute @matmul_tensors
    apply_native_constraint "nestedInFunc"(%0, %1 : !pdl.operation, !pdl.attribute)
    rewrite %0 with "transform.dialect"
  }

  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @pdl_target in %arg1
    %1, %loops1:3 = transform.structured.tile %0 {interchange = [0, 2, 1], sizes = [32, 32, 32]}
    %2, %loops2:3 = transform.structured.tile %1 {interchange = [0, 1, 2], sizes = [4, 4, 1]}
    %3 = transform.structured.pad %2 {padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32], pack_paddings = [1, 1, 1], hoist_paddings = [6, 6, 0], transpose_paddings = [[1, 0], [0, 1]]}
    %4 = transform.structured.vectorize %3  {vectorize_padding = true}
  }
}
