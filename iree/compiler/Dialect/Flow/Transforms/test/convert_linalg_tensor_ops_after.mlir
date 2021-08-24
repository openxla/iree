// RUN: iree-opt -iree-flow-convert-linalg-tensor-ops-pass='run-before-dispatch-region-formation=false' -canonicalize -cse -split-input-file %s | IreeFileCheck %s

func @turn_fill_into_splat(%arg0: tensor<?x?xf32>, %arg1: tensor<f32>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<?x?xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = tensor.extract %arg1[] : tensor<f32>
  %1 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %2 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %3 = affine.apply affine_map<(d0)[s0, s1] -> (d0 + s0 + s1)>(%1)[%arg2, %arg4]
  %4 = affine.apply affine_map<(d0)[s0, s1] -> (d0 + s0 + s1)>(%2)[%arg3, %arg5]
  %5 = linalg.init_tensor [%3, %4] : tensor<?x?xf32>
  %6 = linalg.fill(%0, %5) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
  %7 = flow.tensor.update %arg0, %6[%arg2, %arg3] : tensor<?x?xf32>{%1, %2} -> %6 as tensor<?x?xf32>{%3, %4}
  return %7 : tensor<?x?xf32>
}

//       CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
//       CHECK: func @turn_fill_into_splat
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<f32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG4:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG5:[a-zA-Z0-9]+]]: index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//       CHECK:   %[[VAL:.+]] = tensor.extract %[[ARG1]][]
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[RD0:.+]] = affine.apply #[[MAP]]()[%[[ARG2]], %[[ARG4]], %[[D0]]]
//   CHECK-DAG:   %[[RD1:.+]] = affine.apply #[[MAP]]()[%[[ARG3]], %[[ARG5]], %[[D1]]]
//       CHECK:   %[[SPLAT:.+]] = flow.tensor.splat %[[VAL]] : tensor<?x?xf32>{%[[RD0]], %[[RD1]]}
//       CHECK:   flow.tensor.update %[[ARG0]], %[[SPLAT]]
