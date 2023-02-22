// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-flow-fusion-of-tensor-ops{fuse-multi-use=true}))" %s | FileCheck %s

func.func @softmax(%arg0 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant -3.40282347E+38 : f32
  %1 = tensor.empty() : tensor<12x128xf32>
  %2 = linalg.fill ins(%cst_1 : f32) outs(%1 : tensor<12x128xf32>) -> tensor<12x128xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<12x128x128xf32>) outs(%2 : tensor<12x128xf32>) {
  ^bb0(%b0: f32, %b1: f32):
    %11 = arith.maxf %b0, %b1 : f32
    linalg.yield %11 : f32
  } -> tensor<12x128xf32>
  %4 = tensor.empty() : tensor<12x128x128xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %3 : tensor<12x128x128xf32>, tensor<12x128xf32>) outs(%4 : tensor<12x128x128xf32>) {
  ^bb0(%b0: f32, %b1: f32, %arg2: f32):
    %11 = arith.subf %b0, %b1 : f32
    linalg.yield %11 : f32
  } -> tensor<12x128x128xf32>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<12x128x128xf32>) outs(%4 : tensor<12x128x128xf32>) {
  ^bb0(%b0: f32, %b1: f32):
    %11 = math.exp %b0 : f32
    linalg.yield %11 : f32
  } -> tensor<12x128x128xf32>
  %7 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<12x128xf32>) -> tensor<12x128xf32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6 : tensor<12x128x128xf32>) outs(%7 : tensor<12x128xf32>) {
  ^bb0(%b0: f32, %b1: f32):
    %11 = arith.addf %b0, %b1 : f32
    linalg.yield %11 : f32
  } -> tensor<12x128xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<12x128xf32>) outs(%1 : tensor<12x128xf32>) {
  ^bb0(%b0: f32, %b1: f32):
    %11 = arith.divf %cst, %b0 : f32
    linalg.yield %11 : f32
  } -> tensor<12x128xf32>
  %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %9 : tensor<12x128x128xf32>, tensor<12x128xf32>) outs(%4 : tensor<12x128x128xf32>) {
  ^bb0(%b0: f32, %b1: f32, %arg2: f32):
    %11 = arith.mulf %b0, %b1 : f32
    linalg.yield %11 : f32
  } -> tensor<12x128x128xf32>
  return %10 : tensor<12x128x128xf32>
}
// CHECK-LABEL: func.func @softmax
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<12x128x128xf32>
//       CHECK:   %[[INIT0:.+]] = tensor.empty()
//       CHECK:   %[[FILL0:.+]] = linalg.fill
//  CHECK-SAME:       outs(%[[INIT0]] :
//       CHECK:   %[[GENERIC0:.+]] = linalg.generic
//  CHECK-SAME:       ["parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%[[ARG0]] :
//  CHECK-SAME:       outs(%[[FILL0]] :
//       CHECK:   %[[INIT1:.+]] = tensor.empty()
//       CHECK:   %[[GENERIC1:.+]] = linalg.generic
//  CHECK-SAME:       ["parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[ARG0]], %[[GENERIC0]] :
//  CHECK-SAME:       outs(%[[INIT1]] :
//       CHECK:   %[[FILL1:.+]] = linalg.fill
//  CHECK-SAME:       outs(%[[INIT0]] :
//       CHECK:   %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:       ["parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%[[GENERIC1]] :
//  CHECK-SAME:       outs(%[[FILL1]] :
//       CHECK:   %[[GENERIC3:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[GENERIC1]], %[[GENERIC2]] :
//  CHECK-SAME:       outs(%[[INIT1]] :
//       CHECK:   return %[[GENERIC3]]

// -----

func.func @batchnorm_training(%10 : tensor<12xf32>, %11 : tensor<12x12x12x12x12xf32>, %12 : tensor<12xf32>) -> (tensor<12xf32>, tensor<12xf32>, tensor<12xf32>)
{
  %cst = arith.constant 1.42 : f32
  %cst_1 = arith.constant 1.45 : f32
  %cst_0 = arith.constant 1.3 : f32
  %cst_2 = arith.constant 0.0 : f32
  %13 = tensor.empty() : tensor<12xf32>
  %14 = linalg.fill ins(%cst_2 : f32) outs(%13 : tensor<12xf32>) -> tensor<12xf32>
  %15 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4, d0)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0)>],
      iterator_types = ["parallel", "reduction", "reduction", "reduction", "reduction"]}
      ins(%11, %12 : tensor<12x12x12x12x12xf32>, tensor<12xf32>) outs(%14 : tensor<12xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %19 = arith.subf %arg1, %arg2 : f32
      %20 = arith.mulf %19, %19 : f32
      %21 = arith.addf %arg3, %20 : f32
      linalg.yield %21 : f32
    } -> tensor<12xf32>
  %16 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
      ins(%15: tensor<12xf32>) outs(%13 : tensor<12xf32>) {
    ^bb0(%arg1: f32, %arg2 : f32):
      %19 = arith.divf %arg1, %cst_1 : f32
      %20 = arith.addf %19, %cst_0 : f32
      linalg.yield %20 : f32
    } -> tensor<12xf32>
  %17 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%16 : tensor<12xf32>) outs(%13 : tensor<12xf32>) {
    ^bb0(%arg1: f32, %arg2 : f32):
      %19 = math.sqrt %arg1 : f32
      linalg.yield %19 : f32
    } -> tensor<12xf32>
  %18 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
      {__internal_linalg_transform__ = "tensor_fuse_err"}
      ins(%10, %17 : tensor<12xf32>, tensor<12xf32>) outs(%13 : tensor<12xf32>)  {
    ^bb0(%arg1: f32, %arg2: f32, %arg3 : f32):
      %19 = arith.subf %arg1, %arg2 : f32
      %20 = arith.mulf %19, %cst : f32
      %21 = arith.subf %arg1, %20 : f32
      linalg.yield %21 : f32
    } -> tensor<12xf32>
  return %16, %17, %18 : tensor<12xf32>, tensor<12xf32>, tensor<12xf32>
}
// CHECK-LABEL: func @batchnorm_training(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<12xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<12x12x12x12x12xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<12xf32>
//       CHECK:   %[[INIT:.+]] = tensor.empty() : tensor<12xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:       outs(%[[INIT]] :
//       CHECK:   %[[GENERIC0:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG1]], %[[ARG2]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//       CHECK:   %[[GENERIC1:.+]]:3 = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]], %[[GENERIC0]] :
//  CHECK-SAME:       outs(%[[INIT]], %[[INIT]], %[[INIT]] :
//       CHECK:   return %[[GENERIC1]]#0, %[[GENERIC1]]#1, %[[GENERIC1]]#2

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @fuse_only_with_same_marker(%arg0: tensor<5x5xf32>, %arg1: tensor<5x5xf32>) -> (tensor<5x5xf32>, tensor<5x5xf32>, tensor<5x5xf32>, tensor<5x5xf32>) {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 3.000000e+00 : f32
    %0 = tensor.empty() : tensor<5x5xf32>
    %1 = tensor.empty() : tensor<5x5xf32>
    %2 = tensor.empty() : tensor<5x5xf32>
    %3 = tensor.empty() : tensor<5x5xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<5x5xf32>) outs(%0 : tensor<5x5xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %8 = arith.addf %arg2, %cst : f32
      linalg.yield %8 : f32
    } -> tensor<5x5xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<5x5xf32>) outs(%1 : tensor<5x5xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %8 = arith.subf %arg2, %cst_0 : f32
      linalg.yield %8 : f32
    } -> tensor<5x5xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<5x5xf32>) outs(%2 : tensor<5x5xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %8 = arith.addf %arg2, %cst_1 : f32
      linalg.yield %8 : f32
    } -> tensor<5x5xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %5 : tensor<5x5xf32>, tensor<5x5xf32>) outs(%3 : tensor<5x5xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %8 = arith.subf %arg2, %arg3 : f32
      linalg.yield %8 : f32
    } -> tensor<5x5xf32>
    return %4, %5, %6, %7 : tensor<5x5xf32>, tensor<5x5xf32>, tensor<5x5xf32>, tensor<5x5xf32>
  }
}
// CHECK-LABEL: func.func @fuse_only_with_same_marke
// CHECK:         linalg.generic
// CHECK-NOT:     linalg.generic

// -----

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> ()>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @cleanup_duplicate_input_operands(%arg0: tensor<784x96xf32>) -> tensor<1x28x28x96xf32> {
    %cst = arith.constant 1.000000e+02 : f32
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<f32>
    %cst_1 = arith.constant dense<2.000000e+00> : tensor<f32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<96xf32>
    %0 = tensor.empty() : tensor<784x96xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_2, %arg0 : tensor<96xf32>, tensor<784x96xf32>) outs(%0 : tensor<784x96xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %8 = arith.addf %in, %in_4 : f32
      linalg.yield %8 : f32
    } -> tensor<784x96xf32>
    %expanded = tensor.expand_shape %1 [[0, 1], [2]] : tensor<784x96xf32> into tensor<28x28x96xf32>
    %2 = tensor.empty() : tensor<28x28x96xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %cst_0 : tensor<28x28x96xf32>, tensor<f32>) outs(%2 : tensor<28x28x96xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %8 = arith.addf %in, %in_4 : f32
      linalg.yield %8 : f32
    } -> tensor<28x28x96xf32>
    %4 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3 : tensor<28x28x96xf32>) outs(%2 : tensor<28x28x96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.minf %in, %cst : f32
      linalg.yield %8 : f32
    } -> tensor<28x28x96xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %4 : tensor<28x28x96xf32>, tensor<28x28x96xf32>) outs(%2 : tensor<28x28x96xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %8 = arith.mulf %in, %in_4 : f32
      linalg.yield %8 : f32
    } -> tensor<28x28x96xf32>
    %expanded_3 = tensor.expand_shape %5 [[0, 1], [2], [3]] : tensor<28x28x96xf32> into tensor<1x28x28x96xf32>
    %6 = tensor.empty() : tensor<1x28x28x96xf32>
    %7 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_3 : tensor<1x28x28x96xf32>) outs(%6 : tensor<1x28x28x96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.mulf %in, %in : f32
      linalg.yield %8 : f32
    } -> tensor<1x28x28x96xf32>
    return %7 : tensor<1x28x28x96xf32>
  }
}

// CHECK-LABEL: func.func @cleanup_duplicate_input_operands
