// RUN: iree-dialects-opt --iree-linalg-ext-materialize-encoding -cse -split-input-file %s | FileCheck %s

func.func @pack_unpack_gemm_lhs(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<GEMM_LHS>>
  %1 = iree_linalg_ext.unset_encoding %0 : tensor<?x?xf32, #iree_linalg_ext.encoding<GEMM_LHS>> -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: func @pack_unpack_gemm_lhs(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[OUTER_D0:.+]] = affine.apply #[[MAP0]]()[%[[D0]]]
//  CHECK-DAG:   %[[OUTER_D1:.+]] = affine.apply #[[MAP1]]()[%[[D1]]]
//      CHECK:   %[[PACK_DEST:.+]] = linalg.init_tensor [%[[OUTER_D0]], %[[OUTER_D1]], 8, 4]
//      CHECK:   %[[PACK:.+]] = iree_linalg_ext.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[PACK_DEST]]
//      CHECK:   %[[UNPACK_DEST:.+]] = linalg.init_tensor [%[[D0]], %[[D1]]]
//      CHECK:   %[[UNPACK:.+]] = iree_linalg_ext.unpack %[[PACK]] inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[UNPACK_DEST]]
//      CHECK:   return %[[UNPACK]]

// -----

func.func @pack_unpack_gemm_rhs(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<GEMM_RHS>>
  %1 = iree_linalg_ext.unset_encoding %0 : tensor<?x?xf32, #iree_linalg_ext.encoding<GEMM_RHS>> -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func @pack_unpack_gemm_rhs(
//       CHECK:   linalg_ext.pack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [4, 8]
//       CHECK:   linalg_ext.unpack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [4, 8]

// -----

func.func @pack_unpack_gemm_rhs_transpose(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<GEMM_RHS_TRANSPOSE>>
  %1 = iree_linalg_ext.unset_encoding %0 : tensor<?x?xf32, #iree_linalg_ext.encoding<GEMM_RHS_TRANSPOSE>> -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func @pack_unpack_gemm_rhs_transpose(
//       CHECK:   linalg_ext.pack %{{.+}} outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 4]
//       CHECK:   linalg_ext.unpack %{{.+}} outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 4]

// -----

func.func @pack_unpack_gemm_result(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<GEMM_RESULT>>
  %1 = iree_linalg_ext.unset_encoding %0 : tensor<?x?xf32, #iree_linalg_ext.encoding<GEMM_RESULT>> -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func @pack_unpack_gemm_result(
//       CHECK:   linalg_ext.pack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [8, 8]
//       CHECK:   linalg_ext.unpack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [8, 8]

// -----

func.func @pack_gemm(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250x500xf32>, %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %pad_value = arith.constant 0.0 : f32
  %pad_lhs = tensor.pad %arg0 low[0, 0] high[4, 2] {
    ^bb0(%b0: index, %b1 : index):
      tensor.yield %pad_value : f32
    } : tensor<100x250xf32> to tensor<104x252xf32>
  %lhs = iree_linalg_ext.set_encoding %pad_lhs : tensor<104x252xf32> -> tensor<104x252xf32, #iree_linalg_ext.encoding<GEMM_LHS>>
  %pad_rhs = tensor.pad %arg1 low[0, 0] high[2, 4] {
    ^bb0(%b0: index, %b1 : index):
      tensor.yield %pad_value : f32
    } : tensor<250x500xf32> to tensor<252x504xf32>
  %rhs = iree_linalg_ext.set_encoding %pad_rhs : tensor<252x504xf32> -> tensor<252x504xf32, #iree_linalg_ext.encoding<GEMM_RHS_TRANSPOSE>>
  %pad_output = tensor.pad %arg2 low[0, 0] high[4, 4] {
    ^bb0(%b0: index, %b1 : index):
      tensor.yield %pad_value : f32
    } : tensor<100x500xf32> to tensor<104x504xf32>
  %output = iree_linalg_ext.set_encoding %pad_output : tensor<104x504xf32> -> tensor<104x504xf32, #iree_linalg_ext.encoding<GEMM_RESULT>>
  %gemm_packed = linalg.matmul ins(%lhs, %rhs : tensor<104x252xf32, #iree_linalg_ext.encoding<GEMM_LHS>>, tensor<252x504xf32, #iree_linalg_ext.encoding<GEMM_RHS_TRANSPOSE>>)
      outs(%output : tensor<104x504xf32, #iree_linalg_ext.encoding<GEMM_RESULT>>) -> tensor<104x504xf32, #iree_linalg_ext.encoding<GEMM_RESULT>>
  %gemm = iree_linalg_ext.unset_encoding %gemm_packed : tensor<104x504xf32, #iree_linalg_ext.encoding<GEMM_RESULT>> -> tensor<104x504xf32>
  %result = tensor.extract_slice %gemm[0, 0] [100, 500] [1, 1] : tensor<104x504xf32> to tensor<100x500xf32>
  return %result : tensor<100x500xf32>
}
//      CHECK: func @pack_gemm(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//      CHECK:   %[[CST:.+]] = arith.constant 0.0
//      CHECK:   %[[INIT_LHS:.+]] = linalg.init_tensor [13, 63, 8, 4]
//      CHECK:   %[[PACK_LHS:.+]] = iree_linalg_ext.pack %[[ARG0]] padding_value(%[[CST]] : f32)
// CHECK-SAME:       into %[[INIT_LHS]]
//      CHECK:   %[[INIT_RHS:.+]] = linalg.init_tensor [63, 63, 8, 4]
//      CHECK:   %[[PACK_RHS:.+]] = iree_linalg_ext.pack %[[ARG1]] padding_value(%[[CST]] : f32)
// CHECK-SAME:       into %[[INIT_RHS]]
//      CHECK:   %[[INIT_RESULT:.+]] = linalg.init_tensor [13, 63, 8, 8]
//      CHECK:   %[[PACK_RESULT:.+]] = iree_linalg_ext.pack %[[ARG2]] padding_value(%[[CST]] : f32)
// CHECK-SAME:       into %[[INIT_RESULT]]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
// CHECK-SAME:       outs(%[[PACK_RESULT]] :
//      CHECK:   %[[UNPACK:.+]] = iree_linalg_ext.unpack %[[MMT4D]]
//      CHECK:   return %[[UNPACK]]

// -----

func.func @pack_gemm_dynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<GEMM_LHS>>
  %1 = iree_linalg_ext.set_encoding %arg1 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<GEMM_RHS_TRANSPOSE>>
  %2 = iree_linalg_ext.set_encoding %arg2 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<GEMM_RESULT>>
  %3 = linalg.matmul ins(%0, %1 : tensor<?x?xf32, #iree_linalg_ext.encoding<GEMM_LHS>>, tensor<?x?xf32, #iree_linalg_ext.encoding<GEMM_RHS_TRANSPOSE>>)
      outs(%2 : tensor<?x?xf32, #iree_linalg_ext.encoding<GEMM_RESULT>>) -> tensor<?x?xf32, #iree_linalg_ext.encoding<GEMM_RESULT>>
  %4 = iree_linalg_ext.unset_encoding %3 : tensor<?x?xf32, #iree_linalg_ext.encoding<GEMM_RESULT>> -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: func @pack_gemm_dynamic(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//      CHECK:   %[[PACK_LHS:.+]] = iree_linalg_ext.pack %[[ARG0]]
//      CHECK:   %[[PACK_RHS:.+]] = iree_linalg_ext.pack %[[ARG1]]
//      CHECK:   %[[PACK_RESULT:.+]] = iree_linalg_ext.pack %[[ARG2]]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
// CHECK-SAME:       outs(%[[PACK_RESULT]] :
//      CHECK:   %[[UNPACK:.+]] = iree_linalg_ext.unpack %[[MMT4D]]
//      CHECK:   return %[[UNPACK]]
