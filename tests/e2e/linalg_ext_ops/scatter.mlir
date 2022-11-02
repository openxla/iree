
func.func @scatter_2d_origin() {
  %original = util.unfoldable_constant dense<0> : tensor<2x2xi32>
  %update = util.unfoldable_constant dense<1> : tensor<1xi32>
  %indices = util.unfoldable_constant dense<0> : tensor<1x2xi32>
  %result = iree_linalg_ext.scatter dimension_map(dense<[0, 1]> : tensor<2xi64>) unique_indices(true)
                          ins(%update, %indices : tensor<1xi32>, tensor<1x2xi32>)
                          outs(%original : tensor<2x2xi32>) {
                    ^bb0(%arg0: i32, %arg1: i32):
                      iree_linalg_ext.yield %arg0 : i32
  } -> tensor<2x2xi32>

  check.expect_eq_const(%result, dense<[[1, 0], [0, 0]]> : tensor<2x2xi32>) : tensor<2x2xi32>

  return
}

func.func @scatter_2d_origin_slice_horizontal() {
  %original = util.unfoldable_constant dense<0> : tensor<2x2xi32>
  %update = util.unfoldable_constant dense<1> : tensor<1x2xi32>
  %indices = util.unfoldable_constant dense<0> : tensor<1x2xi32>
  %result = iree_linalg_ext.scatter dimension_map(dense<[0, 1]> : tensor<2xi64>) unique_indices(true)
                          ins(%update, %indices : tensor<1x2xi32>, tensor<1x2xi32>)
                          outs(%original : tensor<2x2xi32>) {
                    ^bb0(%arg0: i32, %arg1: i32):
                      iree_linalg_ext.yield %arg0 : i32
  } -> tensor<2x2xi32>

  check.expect_eq_const(%result, dense<[[1, 1], [0, 0]]> : tensor<2x2xi32>) : tensor<2x2xi32>

  return
}


func.func @scatter_2d_origin_slice_vertical() {
  %original = util.unfoldable_constant dense<0> : tensor<2x2xi32>
  %update = util.unfoldable_constant dense<1> : tensor<1x2x1xi32>
  %indices = util.unfoldable_constant dense<0> : tensor<1x2xi32>
  %result = iree_linalg_ext.scatter dimension_map(dense<[0, 1]> : tensor<2xi64>) unique_indices(true)
                          ins(%update, %indices : tensor<1x2x1xi32>, tensor<1x2xi32>)
                          outs(%original : tensor<2x2xi32>) {
                    ^bb0(%arg0: i32, %arg1: i32):
                      iree_linalg_ext.yield %arg0 : i32
  } -> tensor<2x2xi32>

  check.expect_eq_const(%result, dense<[[1, 0], [1, 0]]> : tensor<2x2xi32>) : tensor<2x2xi32>

  return
}

func.func @scatter_2d_offset() {
  %original = util.unfoldable_constant dense<0> : tensor<2x2xi32>
  %update = util.unfoldable_constant dense<1> : tensor<1xi32>
  %indices = util.unfoldable_constant dense<[[0, 1]]> : tensor<1x2xi32>
  %result = iree_linalg_ext.scatter dimension_map(dense<[0, 1]> : tensor<2xi64>) unique_indices(true)
                          ins(%update, %indices : tensor<1xi32>, tensor<1x2xi32>)
                          outs(%original : tensor<2x2xi32>) {
                    ^bb0(%arg0: i32, %arg1: i32):
                      iree_linalg_ext.yield %arg0 : i32
  } -> tensor<2x2xi32>

  check.expect_eq_const(%result, dense<[[0, 1], [0, 0]]> : tensor<2x2xi32>) : tensor<2x2xi32>

  return
}

func.func @scatter_2d_offset_swapped() {
  %original = util.unfoldable_constant dense<0> : tensor<2x2xi32>
  %update = util.unfoldable_constant dense<1> : tensor<1xi32>
  %indices = util.unfoldable_constant dense<[[0, 1]]> : tensor<1x2xi32>
  %result = iree_linalg_ext.scatter dimension_map(dense<[1, 0]> : tensor<2xi64>) unique_indices(true)
                          ins(%update, %indices : tensor<1xi32>, tensor<1x2xi32>)
                          outs(%original : tensor<2x2xi32>) {
                    ^bb0(%arg0: i32, %arg1: i32):
                      iree_linalg_ext.yield %arg0 : i32
  } -> tensor<2x2xi32>

  check.expect_eq_const(%result, dense<[[0, 0], [1, 0]]> : tensor<2x2xi32>) : tensor<2x2xi32>

  return
}

func.func @scatter_2d_multiple() {
  %original = util.unfoldable_constant dense<0> : tensor<2x2xi32>
  %update = util.unfoldable_constant dense<1> : tensor<2xi32>
  %indices = util.unfoldable_constant dense<[[0, 0], [1, 1]]> : tensor<2x2xi32>
  %result = iree_linalg_ext.scatter dimension_map(dense<[0, 1]> : tensor<2xi64>) unique_indices(true)
                          ins(%update, %indices : tensor<2xi32>, tensor<2x2xi32>)
                          outs(%original : tensor<2x2xi32>) {
                    ^bb0(%arg0: i32, %arg1: i32):
                      iree_linalg_ext.yield %arg0 : i32
  } -> tensor<2x2xi32>

  check.expect_eq_const(%result, dense<[[1, 0], [0, 1]]> : tensor<2x2xi32>) : tensor<2x2xi32>

  return
}

func.func @scatter_2d_multiple_slice() {
  %original = util.unfoldable_constant dense<0> : tensor<3x3xi32>
  %update = util.unfoldable_constant dense<1> : tensor<2x2xi32>
  %indices = util.unfoldable_constant dense<[[0, 1], [1, 0]]> : tensor<2x2xi32>
  %result = iree_linalg_ext.scatter dimension_map(dense<[0, 1]> : tensor<2xi64>) unique_indices(true)
                          ins(%update, %indices : tensor<2x2xi32>, tensor<2x2xi32>)
                          outs(%original : tensor<3x3xi32>) {
                    ^bb0(%arg0: i32, %arg1: i32):
                      iree_linalg_ext.yield %arg0 : i32
  } -> tensor<3x3xi32>

  check.expect_eq_const(%result, dense<[[0, 1, 1], [1, 1, 0], [0, 0, 0]]> : tensor<3x3xi32>) : tensor<3x3xi32>

  return
}

func.func @scatter_2d_multiple_slice_transpose() {
  %original = util.unfoldable_constant dense<0> : tensor<3x4xi32>
  %update = util.unfoldable_constant dense<1> : tensor<2x2xi32>
  %indices = util.unfoldable_constant dense<[[0, 1], [2, 0]]> : tensor<2x2xi32>
  %result = iree_linalg_ext.scatter dimension_map(dense<[1, 0]> : tensor<2xi64>) unique_indices(true)
                          ins(%update, %indices : tensor<2x2xi32>, tensor<2x2xi32>)
                          outs(%original : tensor<3x4xi32>) {
                    ^bb0(%arg0: i32, %arg1: i32):
                      iree_linalg_ext.yield %arg0 : i32
  } -> tensor<3x4xi32>

  check.expect_eq_const(%result, dense<[[0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]]> : tensor<3x4xi32>) : tensor<3x4xi32>

  return
}