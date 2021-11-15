// RUN: iree-opt -split-input-file -canonicalize -cse %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @tensorCastMatchingTypeFolds
func @tensorCastMatchingTypeFolds(%arg0: !hal.buffer_view) -> !hal.buffer_view {
  // CHECK-NOT: hal.tensor.cast
  // CHECK: return %arg0 : !hal.buffer_view
  %0 = hal.tensor.cast %arg0 : !hal.buffer_view -> !hal.buffer_view
  return %0 : !hal.buffer_view
}

// -----

// CHECK-LABEL: @tensorCastPassthroughFolds
func @tensorCastPassthroughFolds(%arg0: !hal.buffer_view) -> !hal.buffer_view {
  // CHECK-NOT: hal.tensor.cast
  // CHECK: return %arg0 : !hal.buffer_view
  %0 = hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<5xi32>
  %1 = hal.tensor.cast %0 : tensor<5xi32> -> !hal.buffer_view
  return %1 : !hal.buffer_view
}

// -----

// CHECK-LABEL: @tensorCastDifferentTypesDoNotFold
func @tensorCastDifferentTypesDoNotFold(%arg0: !hal.buffer_view) -> !hal.buffer {
  // This could be canonicalized if there is only one use.
  // CHECK: %0 = hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<5xi32>
  // CHECK-NEXT: %1 = hal.tensor.cast %0 : tensor<5xi32> -> !hal.buffer
  // CHECK-NEXT: return %1 : !hal.buffer
  %0 = hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<5xi32>
  %1 = hal.tensor.cast %0 : tensor<5xi32> -> !hal.buffer
  return %1 : !hal.buffer
}
