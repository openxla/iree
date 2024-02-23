func.func @normal_conv_1() {
  %0 = util.unfoldable_constant dense<
      [[[[0.5       , 0.5212766 ],
         [0.54255319, 0.56382979],
         [0.58510638, 0.60638298],
         [0.62765957, 0.64893617],
         [0.67021277, 0.69148936],
         [0.71276596, 0.73404255]],

        [[0.75531915, 0.77659574],
         [0.79787234, 0.81914894],
         [0.84042553, 0.86170213],
         [0.88297872, 0.90425532],
         [0.92553191, 0.94680851],
         [0.96808511, 0.9893617 ]],

        [[1.0106383 , 1.03191489],
         [1.05319149, 1.07446809],
         [1.09574468, 1.11702128],
         [1.13829787, 1.15957447],
         [1.18085106, 1.20212766],
         [1.22340426, 1.24468085]],

        [[1.26595745, 1.28723404],
         [1.30851064, 1.32978723],
         [1.35106383, 1.37234043],
         [1.39361702, 1.41489362],
         [1.43617021, 1.45744681],
         [1.4787234 , 1.5       ]]]]> : tensor<1x4x6x2xf32>
  %1 = util.unfoldable_constant dense<
      [[[[0.5       , 0.52857143, 0.55714286],
         [0.58571429, 0.61428571, 0.64285714]],

        [[0.67142857, 0.7       , 0.72857143],
         [0.75714286, 0.78571429, 0.81428571]],

        [[0.84285714, 0.87142857, 0.9       ],
         [0.92857143, 0.95714286, 0.98571429]]],


       [[[1.01428571, 1.04285714, 1.07142857],
         [1.1       , 1.12857143, 1.15714286]],

        [[1.18571429, 1.21428571, 1.24285714],
         [1.27142857, 1.3       , 1.32857143]],

        [[1.35714286, 1.38571429, 1.41428571],
         [1.44285714, 1.47142857, 1.5       ]]]]>
   : tensor<2x3x2x3xf32>
  %2 = "stablehlo.convolution"(%0, %1) {
       batch_group_count = 1 : i64,
       dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0,
          input_feature_dimension = 3,
          input_spatial_dimensions = [1, 2],
          kernel_input_feature_dimension = 2,
          kernel_output_feature_dimension = 3,
          kernel_spatial_dimensions = [0, 1],
          output_batch_dimension = 0,
          output_feature_dimension = 3,
          output_spatial_dimensions = [1, 2]
        >,
       feature_group_count = 1 : i64,
       rhs_dilation = array<i64: 1, 1>,
       window_strides = array<i64: 1, 1>}
       : (tensor<1x4x6x2xf32>, tensor<2x3x2x3xf32>) -> (tensor<1x3x4x3xf32>)
   check.expect_almost_eq_const(%2, dense<
         [[[[ 8.39452888,  8.62796353,  8.86139818],
         [ 8.89057751,  9.13860182,  9.38662614],
         [ 9.38662614,  9.64924012,  9.9118541 ],
         [ 9.88267477, 10.15987842, 10.43708207]],

        [[11.37082067, 11.69179331, 12.01276596],
         [11.8668693 , 12.20243161, 12.53799392],
         [12.36291793, 12.71306991, 13.06322188],
         [12.85896657, 13.22370821, 13.58844985]],

        [[14.34711246, 14.7556231 , 15.16413374],
         [14.84316109, 15.2662614 , 15.6893617 ],
         [15.33920973, 15.7768997 , 16.21458967],
         [15.83525836, 16.28753799, 16.73981763]]]]>
        : tensor<1x3x4x3xf32>) : tensor<1x3x4x3xf32>
   return
}
func.func @normal_conv_2() {
  %input = util.unfoldable_constant dense<
     [[[[6.0, 7.5, 0.0, 1.5],
        [1.5, 3.5, 4.5, 2.0],
        [3.0, 6.0, 0.5, 3.0]],
       [[3.5, 7.0, 2.5, 6.5],
        [4.0, 4.5, 8.0, 2.5],
        [7.5, 7.5, 0.0, 1.5]],
       [[7.0, 3.5, 0.0, 0.5],
        [4.5, 0.0, 5.0, 1.5],
        [5.5, 1.0, 0.0, 0.0]]]]>
    : tensor<1x3x3x4xf32>
  %filter = util.unfoldable_constant dense<
      [[[[2.0, 2.5, 2.5, 3.0, 4.0, 2.0, 0.5, 2.0, 4.5, 5.0, 5.0, 4.0, 0.5, 0.5, 3.5, 4.5,
          4.5, 1.5, 3.0, 3.5, 1.0, 0.0, 1.5, 2.5, 4.5, 5.0, 2.0, 2.0, 3.0, 2.0, 2.0, 1.5],
         [2.0, 2.0, 4.0, 2.0, 1.5, 5.0, 3.5, 2.5, 2.5, 0.0, 0.5, 2.5, 4.5, 1.5, 0.0, 2.5,
          0.0, 0.5, 1.0, 2.0, 1.0, 0.0, 1.5, 1.0, 5.0, 0.0, 3.5, 2.5, 4.5, 0.0, 5.0, 1.0],
         [5.0, 3.5, 1.0, 4.5, 1.0, 1.5, 1.5, 1.0, 1.5, 2.0, 0.5, 1.0, 4.5, 5.0, 0.5, 2.0,
          5.0, 3.0, 4.0, 1.0, 1.5, 0.0, 0.0, 3.0, 0.0, 3.0, 1.5, 5.0, 1.5, 4.0, 4.0, 4.0],
         [1.0, 1.5, 1.0, 0.0, 4.0, 4.0, 1.5, 4.0, 5.0, 1.0, 4.0, 2.0, 1.5, 0.0, 2.0, 1.5,
          3.0, 4.5, 4.0, 0.0, 4.0, 2.5, 4.5, 0.0, 4.5, 3.0, 2.5, 1.5, 0.5, 4.0, 0.0, 2.0]],
        [[4.5, 3.0, 2.5, 3.5, 4.0, 4.0, 4.5, 1.0, 4.0, 3.0, 3.0, 4.5, 0.5, 3.0, 4.0, 4.0,
          1.5, 1.0, 1.5, 5.0, 3.0, 1.5, 3.0, 2.5, 3.5, 0.0, 4.0, 2.0, 5.0, 3.0, 2.5, 4.0],
         [1.0, 1.5, 4.5, 3.5, 2.5, 1.5, 2.0, 2.5, 1.5, 1.5, 3.5, 4.5, 4.5, 4.5, 3.5, 1.5,
          5.0, 1.0, 1.5, 4.5, 5.0, 3.5, 3.5, 2.5, 0.5, 1.0, 1.0, 4.0, 0.5, 2.5, 4.0, 2.0],
         [0.0, 1.0, 2.5, 2.5, 0.0, 4.0, 0.5, 0.5, 0.0, 1.5, 4.0, 4.0, 2.0, 2.0, 0.0, 4.5,
          1.5, 3.5, 1.5, 1.0, 0.5, 0.5, 1.0, 0.5, 2.0, 1.0, 2.5, 2.5, 2.5, 1.0, 2.5, 3.5],
         [3.5, 3.0, 0.5, 3.0, 3.5, 1.0, 1.5, 0.5, 4.5, 2.5, 4.5, 4.5, 1.0, 0.0, 4.5, 0.5,
          4.5, 5.0, 0.0, 3.0, 0.0, 5.0, 2.0, 4.0, 2.0, 1.5, 1.5, 4.0, 4.0, 3.5, 0.0, 1.5]]],
       [[[4.0, 3.5, 3.5, 5.0, 0.5, 4.0, 2.0, 3.5, 0.0, 2.0, 4.5, 0.0, 5.0, 3.0, 2.0, 1.0,
          2.0, 3.0, 1.5, 5.0, 1.5, 3.5, 4.0, 2.5, 0.0, 4.0, 2.5, 2.0, 3.5, 5.0, 5.0, 2.0],
         [0.5, 1.5, 1.5, 4.5, 1.0, 2.5, 1.0, 1.5, 2.5, 5.0, 3.5, 1.0, 3.5, 0.5, 3.0, 5.0,
          2.5, 0.0, 0.0, 5.0, 1.5, 5.0, 0.5, 5.0, 4.5, 4.5, 3.0, 3.0, 3.5, 4.0, 4.0, 3.5],
         [0.0, 4.0, 3.0, 4.0, 4.5, 4.0, 1.5, 3.0, 0.5, 3.5, 2.0, 4.5, 1.0, 0.0, 4.0, 1.0,
          3.5, 4.0, 2.0, 2.0, 0.5, 3.5, 3.0, 4.5, 2.0, 0.5, 2.5, 4.5, 3.5, 0.5, 1.5, 2.5],
         [3.5, 1.5, 3.0, 3.0, 3.5, 4.5, 0.5, 4.5, 3.0, 0.0, 1.5, 4.0, 2.0, 0.5, 2.0, 2.5,
          0.0, 1.5, 5.0, 0.5, 2.0, 2.0, 2.0, 0.0, 0.0, 5.0, 4.0, 2.0, 3.0, 4.5, 1.5, 1.5]],
        [[1.0, 0.5, 5.0, 1.0, 0.5, 1.5, 2.0, 5.0, 0.5, 0.5, 0.0, 3.5, 4.0, 5.0, 2.0, 1.5,
          2.5, 3.0, 1.5, 1.0, 4.5, 4.0, 0.5, 2.0, 5.0, 0.0, 4.0, 1.5, 4.5, 2.5, 2.5, 0.5],
         [3.5, 4.0, 3.0, 2.0, 3.5, 1.5, 2.5, 1.5, 3.0, 2.0, 3.5, 1.5, 0.0, 2.5, 4.5, 1.5,
          3.5, 2.5, 2.5, 4.0, 0.0, 4.0, 1.5, 3.0, 4.5, 5.0, 1.5, 1.0, 3.5, 0.0, 1.5, 5.0],
         [0.0, 1.5, 3.0, 0.5, 4.5, 1.0, 4.5, 2.0, 4.5, 0.5, 1.5, 1.0, 2.0, 4.5, 3.5, 2.0,
          4.5, 2.0, 0.5, 1.0, 3.5, 1.0, 1.5, 4.5, 5.0, 3.5, 5.0, 3.0, 3.0, 1.0, 5.0, 1.5],
         [3.0, 0.0, 5.0, 4.0, 0.0, 5.0, 3.5, 3.0, 2.5, 4.5, 3.0, 2.5, 1.0, 3.5, 0.5, 4.5,
          1.0, 1.0, 2.5, 3.0, 2.0, 1.0, 1.0, 0.5, 0.0, 4.5, 0.0, 1.0, 4.0, 1.5, 5.0, 0.0]]]]>
    : tensor<2x2x4x32xf32>

    %0 = "stablehlo.convolution"(%input, %filter) {batch_group_count = 1 : i64,
      dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = array<i64: 1, 1>, window_strides = array<i64: 1, 1>} : (tensor<1x3x3x4xf32>, tensor<2x2x4x32xf32>) -> tensor<1x2x2x32xf32>

   check.expect_almost_eq_const(%0, dense<
     [[[[113.25, 127.0, 198.0, 173.25, 159.5, 190.75, 135.5, 160.0,
         169.5, 130.0, 173.75, 174.5, 158.5, 136.75, 159.75, 177.75,
         164.5, 122.25, 116.0, 168.0, 124.75, 144.0, 113.5, 159.0,
         208.0, 186.5, 190.5, 158.5, 213.75, 140.5, 206.75, 135.25],
        [129.75, 147.25, 181.25, 181.75, 142.5, 161.75, 117.75, 153.25,
         119.5, 128.75, 149.25, 171.0, 152.5, 142.5, 166.0, 122.25,
         177.75, 142.75, 116.5, 170.0, 117.5, 176.75, 116.75, 162.25,
         161.25, 135.0, 145.5, 163.25, 190.5, 138.25, 162.5, 146.75]],
       [[111.75, 115.75, 173.5, 158.25, 122.5, 187.25, 129.0, 142.5,
         142.25, 109.0, 175.75, 158.5, 172.75, 146.25, 122.25, 157.25,
         157.5, 141.25, 104.25, 151.25, 136.25, 122.0, 127.75, 125.75,
         180.5, 131.25, 168.75, 151.5, 180.75, 152.75, 193.5, 128.75],
        [138.25, 133.75, 157.5, 168.5, 131.0, 149.75, 115.25, 130.75,
         114.5, 107.25, 127.75, 163.75, 153.5, 149.25, 133.5, 114.0,
         164.75, 120.75, 116.0, 149.5, 127.5, 113.5, 116.0, 129.75,
         126.75, 94.25, 135.0, 157.75, 158.75, 142.0, 158.75, 126.25]]]]>
        : tensor<1x2x2x32xf32>) : tensor<1x2x2x32xf32>
   return
}

func.func @depthwise_conv() {
  %input = util.unfoldable_constant dense<
    [[[[6.0, 7.5, 0.0, 1.5, 1.5, 3.5, 4.5, 2.0, 3.0, 6.0, 0.5, 3.0, 3.5, 7.0, 2.5, 6.5],
       [4.0, 4.5, 8.0, 2.5, 7.5, 7.5, 0.0, 1.5, 7.0, 3.5, 0.0, 0.5, 4.5, 0.0, 5.0, 1.5],
       [5.5, 1.0, 0.0, 0.0, 2.0, 2.5, 3.0, 4.0, 7.5, 2.0, 4.5, 5.0, 0.5, 0.5, 3.5, 4.5],
       [1.5, 3.0, 5.5, 7.0, 0.0, 7.0, 1.5, 6.0, 5.0, 5.5, 2.0, 3.0, 2.0, 7.5, 1.5, 6.0]]]]>
    : tensor<1x1x4x16xf32>
  %filter = util.unfoldable_constant dense<
    [[[[2.0, 2.0, 4.0, 2.0, 1.5, 5.0, 3.5, 2.5, 2.5, 0.0, 0.5, 2.5, 4.5, 1.5, 0.0, 2.5]]]]>
    : tensor<1x1x1x16xf32>

    %0 = "stablehlo.convolution"(%input, %filter) {batch_group_count = 1 : i64,
      dimension_numbers = #stablehlo.conv<raw
        input_batch_dimension = 0,
        input_feature_dimension = 3,
        input_spatial_dimensions = [1, 2],
        kernel_input_feature_dimension = 2,
        kernel_output_feature_dimension = 3,
        kernel_spatial_dimensions = [0, 1],
        output_batch_dimension = 0,
        output_feature_dimension = 3,
        output_spatial_dimensions = [1, 2]
      >, feature_group_count = 16 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = array<i64: 1, 1>, window_strides = array<i64: 1, 1>} : (tensor<1x1x4x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x4x16xf32>

   check.expect_almost_eq_const(%0, dense<
     [[[[12.0, 15.0, 0.0, 3.0, 2.25, 17.5, 15.75, 5.0, 7.5, 0.0, 0.25, 7.5, 15.75, 10.5, 0.0, 16.25],
        [8.0, 9.0, 32.0, 5.0, 11.25, 37.5, 0.0, 3.75, 17.5, 0.0, 0.0, 1.25, 20.25, 0.0, 0.0, 3.75],
        [11.0, 2.0, 0.0, 0.0, 3.0, 12.5, 10.5, 10.0, 18.75, 0.0, 2.25, 12.5, 2.25, 0.75, 0.0, 11.25],
        [3.0, 6.0, 22.0, 14.0, 0.0, 35.0, 5.25, 15.0, 12.5, 0.0, 1.0, 7.5, 9.0, 11.25, 0.0, 15.0]]]]>
        : tensor<1x1x4x16xf32>) : tensor<1x1x4x16xf32>
   return
}

