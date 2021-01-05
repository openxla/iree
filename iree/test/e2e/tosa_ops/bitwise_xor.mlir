func @tensor() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[0, 3, 13, 7]> : tensor<4xi32>
  %1 = iree.unfoldable_constant dense<[0, 2, 7, 0]> : tensor<4xi32>
  %result = "tosa.bitwise_xor"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[0, 1, 10, 7]> : tensor<4xi32>) : tensor<4xi32>
  return
}
