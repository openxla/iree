// RUN: iree-opt -iree-transform-dialect-interpreter --split-input-file %s --verify-diagnostics

#layout = #iree_vector_ext.layout<<[VECTORY], [16]>, <[VECTORX], [16]>>

// Propagate the layout from transfer_read to everyone.
builtin.module attributes { transform.with_named_sequence } {
  func.func @propagate_simple(%arr: memref<16x16xf16>, %a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true], "__vector_layout_test_anchor_result_0" = #layout} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %c = arith.mulf %root, %b : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %d = arith.addf %c, %a : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    func.return %d : vector<16x16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.layout<<[VECTORY], [16]>, <[VECTORX], [16]>>

// Enforce the layout from the transfer_write to everyone
builtin.module attributes { transform.with_named_sequence } {
  func.func @enforce_simple(%arr: memref<16x16xf16>, %a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0 = arith.constant dense<0.0> : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %c = arith.mulf %cst0, %b : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %d = arith.addf %c, %a : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    vector.transfer_write %d, %arr[%c0, %c0] {in_bounds = [true, true], "__vector_layout_test_anchor_operand_0" = #layout} : vector<16x16xf16>, memref<16x16xf16>
    func.return %d : vector<16x16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.layout<<[VECTORY], [16]>, <[VECTORX], [16]>>

// First propagate the layout, and then enforce it up.
builtin.module attributes { transform.with_named_sequence } {
  func.func @propagate_and_enforce(%arr: memref<16x16xf16>, %arr2: memref<16x16xf16>, %a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true], "__vector_layout_test_anchor_result_0" = #layout} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %root2 = vector.transfer_read %arr2[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %c = arith.mulf %root, %b : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %d = arith.addf %c, %a : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %e = arith.divf %d, %root2 : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    func.return %e : vector<16x16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.layout<<[VECTORY], [16]>, <[BATCHY, VECTORX], [2, 8]>>

// Propagate and enforce through reduction.
builtin.module attributes { transform.with_named_sequence } {
  func.func @reduction(%arr: memref<16x16xf16>, %arr2: memref<16xf16>, %a: vector<16xf16>, %b: vector<16xf16>) -> vector<16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0_1 = arith.constant dense<0.0> : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>>}}
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true], "__vector_layout_test_anchor_result_0" = #layout} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ BATCHY,  VECTORX], [2, 8]>>}}
    %root2 = vector.transfer_read %arr2[%c0], %cst_0 {in_bounds = [true]} : memref<16xf16>, vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>>}}
    %root_red = vector.multi_reduction<add>, %root, %cst0_1 [0]  : vector<16x16xf16> to vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>>}}
    %c = arith.mulf %root_red, %b : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>>}}
    %d = arith.addf %c, %a : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>>}}
    %e = arith.divf %d, %root2 : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>>}}
    func.return %e : vector<16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.layout<<[VECTORY], [16]>, <[BATCHY, VECTORX], [2, 8]>>

// Propagate and enforce through transpose and then reduction.
builtin.module attributes { transform.with_named_sequence } {
  func.func @transpose_and_reduction(%arr: memref<16x16xf16>, %arr2: memref<16xf16>, %a: vector<16xf16>, %b: vector<16xf16>) -> vector<16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0_1 = arith.constant dense<0.0> : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true], "__vector_layout_test_anchor_result_0" = #layout} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ BATCHY,  VECTORX], [2, 8]>>}}
    %root2 = vector.transfer_read %arr2[%c0], %cst_0 {in_bounds = [true]} : memref<16xf16>, vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
    %root_transpose = vector.transpose %root, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>, <[ VECTORY], [16]>>}}
    %root_red = vector.multi_reduction<add>, %root_transpose, %cst0_1 [0]  : vector<16x16xf16> to vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
    %c = arith.mulf %root_red, %b : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
    %d = arith.addf %c, %a : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
    %e = arith.divf %d, %root2 : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
    func.return %e : vector<16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layoutA = #iree_vector_ext.layout<<[VECTORX], [32]>, <[VECTORY], [64]>>
#layoutB = #iree_vector_ext.layout<<[VECTORX], [128]>, <[VECTORY], [64]>>
#layoutC = #iree_vector_ext.layout<<[VECTORY], [128]>, <[VECTORX], [32]>>

#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d0)>

// Propagate through vector.contract.
builtin.module attributes { transform.with_named_sequence } {
  func.func @contract(%A : vector<32x64xf16>, %B : vector<128x64xf16>, %C : vector<128x32xf32>) -> vector<128x32xf32> {
    // Check if the layout of %C was properly propagated to %D.
    // expected-remark @below {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [128]>, <[ VECTORX], [32]>>}}
    %D = vector.contract
        {indexing_maps = [#map1, #map2, #map3],
         iterator_types = ["parallel", "parallel", "reduction"],
         kind = #vector.kind<add>,
         "__vector_layout_test_anchor_operand_0" = #layoutB,
         "__vector_layout_test_anchor_operand_1" = #layoutA,
         "__vector_layout_test_anchor_operand_2" = #layoutC
        } %B, %A, %C : vector<128x64xf16>, vector<32x64xf16> into vector<128x32xf32>
    func.return %D : vector<128x32xf32>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.layout<<[VECTORY], [16]>, <[BATCHY, VECTORX], [2, 8]>>

// Propagate and enforce through scf.for
builtin.module attributes { transform.with_named_sequence } {
  func.func @scffor(%arr: memref<16x16xf16>, %arr2: memref<16xf16>, %a: vector<16xf16>, %b: vector<16xf16>) -> vector<16xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0_1 = arith.constant dense<0.0> : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}

    %out = scf.for %iv = %c0 to %c1024 step %c1 iter_args(%arg1 = %cst0_1) -> (vector<16xf16>) {
      // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
      %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true], "__vector_layout_test_anchor_result_0" = #layout} : memref<16x16xf16>, vector<16x16xf16>
      // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ BATCHY,  VECTORX], [2, 8]>>}}
      %root2 = vector.transfer_read %arr2[%c0], %cst_0 {in_bounds = [true]} : memref<16xf16>, vector<16xf16>
      // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
      %root_transpose = vector.transpose %root, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
      // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>, <[ VECTORY], [16]>>}}
      %root_red = vector.multi_reduction<add>, %root_transpose, %arg1 [0]  : vector<16x16xf16> to vector<16xf16>
      // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
      %c = arith.mulf %root_red, %b : vector<16xf16>
      // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
      %d = arith.addf %c, %a : vector<16xf16>
      // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
      %e = arith.divf %d, %root2 : vector<16xf16>
      // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
      scf.yield %e : vector<16xf16>
    }

    func.return %out : vector<16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup = [1, 1],
  outers_per_batch = [1, 1],
  threads_per_outer = [4, 16],
  elements_per_thread = [4, 1],

  subgroup_basis = [1, 1],
  thread_basis   = [4, 16]
>

// Propagate and enforce through reduction along dim 0.
// The printing of this layout is too long for a remark. Just verify that
// the subgroup/thread bases are what we expect.
builtin.module attributes { transform.with_named_sequence } {
  func.func @reduction(%arr: memref<16x16xf16>, %arr2: memref<16xf16>, %a: vector<16xf16>, %b: vector<16xf16>) -> vector<16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0_1 = arith.constant dense<0.0> : vector<16xf16>
    // expected-remark @above {{subgroup_basis = [1, 1], subgroup_active_ids = [false, true], thread_basis = [4, 16], thread_active_ids = [false, true]}}
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true], "__vector_layout_test_anchor_result_0" = #layout} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{thread_basis = [4, 16]}}
    %root_red = vector.multi_reduction<add>, %root, %cst0_1 [0]  : vector<16x16xf16> to vector<16xf16>
    // expected-remark @above {{subgroup_basis = [1, 1], subgroup_active_ids = [false, true], thread_basis = [4, 16], thread_active_ids = [false, true]}}
    %c = arith.mulf %root_red, %a : vector<16xf16>
    // expected-remark @above {{subgroup_basis = [1, 1], subgroup_active_ids = [false, true], thread_basis = [4, 16], thread_active_ids = [false, true]}}
    func.return %c : vector<16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup = [1, 1],
  outers_per_batch = [1, 1],
  threads_per_outer = [4, 16],
  elements_per_thread = [4, 1],

  subgroup_basis = [1, 1],
  thread_basis   = [4, 16]
>

// Propagate and enforce through reduction along dim 1.
// The printing of this layout is too long for a remark. Just verify that
// the subgroup/thread bases are what we expect.
builtin.module attributes { transform.with_named_sequence } {
  func.func @reduction(%arr: memref<16x16xf16>, %arr2: memref<16xf16>, %a: vector<16xf16>, %b: vector<16xf16>) -> vector<16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0_1 = arith.constant dense<0.0> : vector<16xf16>
    // expected-remark @above {{subgroup_basis = [1, 1], subgroup_active_ids = [true, false], thread_basis = [4, 16], thread_active_ids = [true, false]}}
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true], "__vector_layout_test_anchor_result_0" = #layout} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{thread_basis = [4, 16]}}
    %root_red = vector.multi_reduction<add>, %root, %cst0_1 [1]  : vector<16x16xf16> to vector<16xf16>
    // expected-remark @above {{subgroup_basis = [1, 1], subgroup_active_ids = [true, false], thread_basis = [4, 16], thread_active_ids = [true, false]}}
    %c = arith.mulf %root_red, %a : vector<16xf16>
    // expected-remark @above {{subgroup_basis = [1, 1], subgroup_active_ids = [true, false], thread_basis = [4, 16], thread_active_ids = [true, false]}}
    func.return %c : vector<16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [2, 1, 1],
  batches_per_subgroup = [1, 2, 4],
  outers_per_batch = [1, 1, 1],
  threads_per_outer = [4, 8, 2],
  elements_per_thread = [4, 1, 2],

  subgroup_basis = [2, 1, 1],
  thread_basis   = [4, 8, 2]
>

// Propagate and enforce through reduction along dim 1.
// The printing of this layout is too long for a remark. Just verify that
// the subgroup/thread bases are what we expect.
builtin.module attributes { transform.with_named_sequence } {
  func.func @transpose_3d(%arr: memref<32x32x32xf16>) -> () {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0_1 = arith.constant dense<0.0> : vector<16xf16>
    // expected-remark-re @below {{{{.*}}[2, 1, 1], {{.*}}[1, 2, 4], {{.*}}[1, 1, 1], {{.*}}[4, 8, 2], {{.*}}[4, 1, 2]}}
    %root = vector.transfer_read %arr[%c0, %c0, %c0], %cst_0 {
      in_bounds = [true, true, true],
      "__vector_layout_test_anchor_result_0" = #layout
    } : memref<32x32x32xf16>, vector<32x16x16xf16>
    %t = vector.transpose %root, [1, 2, 0] : vector<32x16x16xf16> to vector<16x16x32xf16>
    // expected-remark-re @above {{{{.*}}[1, 1, 2], {{.*}}[2, 4, 1], {{.*}}[1, 1, 1], {{.*}}[8, 2, 4], {{.*}}[1, 2, 4], {{.*}}[2, 0, 1], {{.*}}[2, 0, 1]}}
    vector.transfer_write %t, %arr[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<16x16x32xf16>, memref<32x32x32xf16>
    func.return
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup = [4, 1],
  outers_per_batch = [1, 1],
  threads_per_outer = [32, 4],
  elements_per_thread = [1, 32],
  subgroup_basis = [1, 1],
  thread_basis = [32, 4]
>

// Propagate and enforce layout through broadcast transpose and broadcast.
// Main thing we want to see here is the subgroup_active_ids and thread_active_ids
// are being determined properly.
builtin.module attributes { transform.with_named_sequence } {
  func.func @broadcast_transpose(%quant :  memref<128x128xi4>, %scale : memref<128xf16>, %arr: memref<128x128xf16>) -> () {
    %cst = arith.constant 0.000000e+00 : f16
    %c0_i4 = arith.constant 0 : i4
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %quant[%c0, %c0], %c0_i4 {in_bounds = [true, true], "__vector_layout_test_anchor_result_0" = #layout} : memref<128x128xi4>, vector<128x128xi4>
    // expected-remark @above {{threads_per_outer = [32, 4], elements_per_thread = [1, 32], subgroup_basis = [1, 1], thread_basis = [32, 4]}}
    %1 = vector.transfer_read %scale[%c0],   %cst {in_bounds = [true]} : memref<128xf16>, vector<128xf16>
    // expected-remark @above {{subgroup_active_ids = [true, false], thread_basis = [32, 4], thread_active_ids = [true, false]}}
    %2 = vector.broadcast %1 : vector<128xf16> to vector<128x128xf16>
    // expected-remark-re @above {{threads_per_outer = [4, 32], elements_per_thread = [32, 1], subgroup_order = [1, 0], {{.*}}thread_order = [1, 0], {{.*}}thread_basis = [32, 4]}}
    %3 = vector.transpose %2, [1, 0] : vector<128x128xf16> to vector<128x128xf16>
    // expected-remark @above {{threads_per_outer = [32, 4], elements_per_thread = [1, 32], subgroup_basis = [1, 1], thread_basis = [32, 4]}}
    %4 = arith.extui %0 : vector<128x128xi4> to vector<128x128xi32>
    // expected-remark @above {{threads_per_outer = [32, 4], elements_per_thread = [1, 32], subgroup_basis = [1, 1], thread_basis = [32, 4]}}
    %5 = arith.uitofp %4 : vector<128x128xi32> to vector<128x128xf16>
    // expected-remark @above {{threads_per_outer = [32, 4], elements_per_thread = [1, 32], subgroup_basis = [1, 1], thread_basis = [32, 4]}}
    %6 = arith.mulf %5, %3 : vector<128x128xf16>
    // expected-remark @above {{threads_per_outer = [32, 4], elements_per_thread = [1, 32], subgroup_basis = [1, 1], thread_basis = [32, 4]}}
    vector.transfer_write %6, %arr[%c0, %c0] {in_bounds = [true, true]} : vector<128x128xf16>, memref<128x128xf16>
    func.return
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [2, 1, 1],
  batches_per_subgroup = [1, 2, 4],
  outers_per_batch = [1, 1, 1],
  threads_per_outer = [4, 8, 2],
  elements_per_thread = [4, 1, 2],

  subgroup_basis = [2, 1, 1],
  thread_basis   = [4, 8, 2]
>

/// Invalid anchor tests

// Rank mismatch anchor.
builtin.module attributes { transform.with_named_sequence } {
  // expected-note @below {{when applied to this op}}
  func.func @invalid_rank_nested_layout_anchor(%a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
    %c = arith.addf %a, %b : vector<16x16xf16>
    // expected-error @above {{Rank of vector (2) does not match rank of layout (3)}}
    func.return {"__vector_layout_test_anchor_operand_0" = #layout} %c : vector<16x16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    // expected-error @above {{transform.iree.test_vector_layout_analysis failed to apply}}
    transform.yield
  }
}

// -----

#layout2 = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup = [2, 4],
  outers_per_batch = [1, 1],
  threads_per_outer = [8, 2],
  elements_per_thread = [2, 2],

  subgroup_basis = [1, 1],
  thread_basis   = [8, 2]
>

// Size mismatch anchor.
builtin.module attributes { transform.with_named_sequence } {
  // expected-note @below {{when applied to this op}}
  func.func @invalid_size_nested_layout_anchor(%a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
    %c = arith.addf %a, %b : vector<16x16xf16>
    // expected-error @above {{Vector shape: [16, 16] does not match the layout (nested_layout<subgroups_per_workgroup = [1, 1], batches_per_subgroup = [2, 4], outers_per_batch = [1, 1], threads_per_outer = [8, 2], elements_per_thread = [2, 2], subgroup_basis = [1, 1], thread_basis = [8, 2]>) at dim 0. Dimension expected by layout: 32 actual: 16}}
    func.return {"__vector_layout_test_anchor_operand_0" = #layout2} %c : vector<16x16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    // expected-error @above {{transform.iree.test_vector_layout_analysis failed to apply}}
    transform.yield
  }
}
