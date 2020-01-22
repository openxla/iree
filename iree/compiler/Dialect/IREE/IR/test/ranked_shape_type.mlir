// RUN: iree-opt -split-input-file -verify-diagnostics %s | IreeFileCheck %s

// CHECK-LABEL: @parseScalarShape
// CHECK: !iree.ranked_shape<i32>
func @parseScalarShape(%arg0 : !iree.ranked_shape<i32>) {
  return
}

// -----
// CHECK-LABEL: @parseStaticShape
// CHECK: !iree.ranked_shape<1x2xi32>
func @parseStaticShape(%arg0 : !iree.ranked_shape<1x2xi32>) {
  return
}

// -----
// CHECK-LABEL: @parseDynamicShape
// CHECK: !iree.ranked_shape<1x?x2x?xi32>
func @parseDynamicShape(%arg0 : !iree.ranked_shape<1x?x2x?xi32>) {
  return
}

// -----
// expected-error @+1 {{RankedShapeType must have an integral dim type}}
func @error(%arg0 : !iree.ranked_shape<1x?xf32>) {
  return
}

// -----
func @static_ranked_shape_validation() {
  %0 = iree.static_ranked_shape -> !iree.ranked_shape<1x2xi32>
  // expected-error @+1 {{op must return a fully static ranked_shape (got '!iree.ranked_shape<1x?xi32>')}}
  %1 = iree.static_ranked_shape -> !iree.ranked_shape<1x?xi32>
  return
}

// -----
func @get_ranked_shape_same_rank(%arg0 : tensor<2x?x4xf32>) {
  // expected-error @+1 {{op operand and result must be of same rank}}
  %0 = iree.get_ranked_shape %arg0 : tensor<2x?x4xf32> -> !iree.ranked_shape<2xi32>
  return
}

// -----
func @get_ranked_shape_not_equal_dims(%arg0 : tensor<2x?x4xf32>) {
  // expected-error @+1 {{op operand tensor and result shape must be equal}}
  %0 = iree.get_ranked_shape %arg0 : tensor<2x?x4xf32> -> !iree.ranked_shape<2x2x4xi32>
  return
}
