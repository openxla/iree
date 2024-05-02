// RUN: iree-dialects-opt --split-input-file %s | FileCheck %s

#row_layout1 = #iree_vector_ext.per_dim_layout<[BATCHX, LANEX, VECTORY], [2, 4, 4]>
#col_layout1 = #iree_vector_ext.per_dim_layout<[BATCHY, LANEY, VECTORX], [4, 2, 4]>
#layout1 = #iree_vector_ext.layout<#row_layout1, #col_layout1>
#layout2 = #iree_vector_ext.layout<#col_layout1, #row_layout1>
func.func @specify_layout(%lhs: memref<32x32xf16>) -> vector<32x32xf16> {
  %cst_0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %result = vector.transfer_read %lhs[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x32xf16>, vector<32x32xf16>
  %2 = iree_vector_ext.layout_conflict_resolution %result {sourceLayout = #layout1, desiredLayout = #layout2} : vector<32x32xf16> -> vector<32x32xf16>
  return %2 : vector<32x32xf16>
}

// CHECK-DAG: #[[LAYOUT0:.+]] = #iree_vector_ext.layout<<[ BATCHY,  LANEY,  VECTORX], [4, 2, 4]>, <[ BATCHX,  LANEX,  VECTORY], [2, 4, 4]>>
// CHECK-DAG: #[[LAYOUT1:.+]] = #iree_vector_ext.layout<<[ BATCHX,  LANEX,  VECTORY], [2, 4, 4]>, <[ BATCHY,  LANEY,  VECTORX], [4, 2, 4]>>
// CHECK-LABEL: func.func @specify_layout
// CHECK:      iree_vector_ext.layout_conflict_resolution
// CHECK-SAME:         desiredLayout = #[[LAYOUT0]]
// CHECK-SAME:         sourceLayout = #[[LAYOUT1]]

// -----

#nested_1 = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup = [2, 4],
  outers_per_batch = [4, 1],
  threads_per_outer = [4, 2],
  elements_per_thread = [1, 4],

  subgroup_order = [0, 1],
  thread_order = [0, 1],

  subgroup_basis = [1, 1],
  thread_basis   = [4, 2]
>

#nested_2 = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup = [4, 2],
  outers_per_batch = [1, 4],
  threads_per_outer = [2, 4],
  elements_per_thread = [4, 1],

  subgroup_order = [1, 0],
  thread_order = [1, 0],

  subgroup_basis = [1, 1],
  thread_basis   = [2, 4]
>

#nested_3 = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup = [4, 2],
  outers_per_batch = [1, 4],
  threads_per_outer = [2, 4],
  elements_per_thread = [4, 1],

  subgroup_order = [1, 0],
  thread_order = [1, 0],

  subgroup_basis = [2, 4, 8],
  subgroup_active_ids = [true, true, false],
  thread_basis   = [2, 4]
>

#nested_4 = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup = [4, 2],
  outers_per_batch = [1, 4],
  threads_per_outer = [2, 4],
  elements_per_thread = [4, 1],

  subgroup_order = [1, 0],
  thread_order = [1, 0],

  subgroup_basis = [2, 4, 8],
  subgroup_active_ids = [true, true, false],
  thread_basis   = [2, 4, 2],
  thread_active_ids = [false, true, true]
>

#nested_5 = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup = [4, 2],
  outers_per_batch = [1, 4],
  threads_per_outer = [2, 4],
  elements_per_thread = [4, 1],

  subgroup_order = [1, 0],
  thread_order = [1, 0],

  subgroup_basis = [2, 4],
  subgroup_active_ids = [true, true],
  thread_basis   = [4, 2],
  thread_active_ids = [true, true]
>

func.func @specify_nested(%lhs: memref<32x32xf16>) -> vector<32x32xf16> {
  %cst_0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %result = vector.transfer_read %lhs[%c0, %c0], %cst_0 {
    in_bounds = [true, true],
    layout0 = #nested_1,
    layout1 = #nested_2,
    layout2 = #nested_3,
    layout3 = #nested_4,
    layout4 = #nested_5
  } : memref<32x32xf16>, vector<32x32xf16>
  return %result : vector<32x32xf16>
}

// CHECK: #[[LAYOUT0:.+]] = #iree_vector_ext.nested_layout<
// CHECK-SAME: subgroups_per_workgroup = [1, 1],
// CHECK-SAME: batches_per_subgroup = [2, 4],
// CHECK-SAME: outers_per_batch = [4, 1],
// CHECK-SAME: threads_per_outer = [4, 2],
// CHECK-SAME: elements_per_thread = [1, 4],
// CHECK-SAME: subgroup_basis = [1, 1],
// CHECK-SAME: thread_basis = [4, 2]>

// CHECK: #[[LAYOUT1:.+]] = #iree_vector_ext.nested_layout<
// CHECK-SAME: subgroups_per_workgroup = [1, 1],
// CHECK-SAME: batches_per_subgroup = [4, 2],
// CHECK-SAME: outers_per_batch = [1, 4],
// CHECK-SAME: threads_per_outer = [2, 4],
// CHECK-SAME: elements_per_thread = [4, 1],
// CHECK-SAME: subgroup_order = [1, 0],
// CHECK-SAME: thread_order = [1, 0],
// CHECK-SAME: subgroup_basis = [1, 1],
// CHECK-SAME: thread_basis = [2, 4]>

// CHECK: #[[LAYOUT2:.+]] = #iree_vector_ext.nested_layout<
// CHECK-SAME: subgroups_per_workgroup = [1, 1],
// CHECK-SAME: batches_per_subgroup = [4, 2],
// CHECK-SAME: outers_per_batch = [1, 4],
// CHECK-SAME: threads_per_outer = [2, 4],
// CHECK-SAME: elements_per_thread = [4, 1],
// CHECK-SAME: subgroup_order = [1, 0],
// CHECK-SAME: thread_order = [1, 0],
// CHECK-SAME: subgroup_basis = [2, 4, 8],
// CHECK-SAME: subgroup_active_ids = [true, true, false],
// CHECK-SAME: thread_basis = [2, 4]>

// CHECK: #[[LAYOUT3:.+]] = #iree_vector_ext.nested_layout<
// CHECK-SAME: subgroups_per_workgroup = [1, 1],
// CHECK-SAME: batches_per_subgroup = [4, 2],
// CHECK-SAME: outers_per_batch = [1, 4],
// CHECK-SAME: threads_per_outer = [2, 4],
// CHECK-SAME: elements_per_thread = [4, 1],
// CHECK-SAME: subgroup_order = [1, 0],
// CHECK-SAME: thread_order = [1, 0],
// CHECK-SAME: subgroup_basis = [2, 4, 8],
// CHECK-SAME: subgroup_active_ids = [true, true, false],
// CHECK-SAME: thread_basis = [2, 4, 2],
// CHECK-SAME: thread_active_ids = [false, true, true]>

// CHECK: #[[LAYOUT4:.+]] = #iree_vector_ext.nested_layout<
// CHECK-SAME: subgroups_per_workgroup = [1, 1],
// CHECK-SAME: batches_per_subgroup = [4, 2],
// CHECK-SAME: outers_per_batch = [1, 4],
// CHECK-SAME: threads_per_outer = [2, 4],
// CHECK-SAME: elements_per_thread = [4, 1],
// CHECK-SAME: subgroup_order = [1, 0],
// CHECK-SAME: thread_order = [1, 0],
// CHECK-SAME: subgroup_basis = [2, 4],
// CHECK-SAME: thread_basis = [4, 2]>

// CHECK-LABEL: func.func @specify_nested
// CHECK:      vector.transfer_read
// CHECK-SAME:         layout0 = #[[LAYOUT0]]
// CHECK-SAME:         layout1 = #[[LAYOUT1]]
// CHECK-SAME:         layout2 = #[[LAYOUT2]]
// CHECK-SAME:         layout3 = #[[LAYOUT3]]
// CHECK-SAME:         layout4 = #[[LAYOUT4]]

// -----

func.func @to_simd_op(%simt: vector<4x4x4xf16>) -> vector<64x64xf16> {
  %simd = iree_vector_ext.to_simd %simt : vector<4x4x4xf16> -> vector<64x64xf16>
  func.return %simd : vector<64x64xf16>
}
// CHECK-LABEL: func.func @to_simd_op
// CHECK:      iree_vector_ext.to_simd

// -----

func.func @to_simt_op(%simd: vector<64x64xf32>) -> vector<4x4x4xf32> {
  %simt = iree_vector_ext.to_simd %simd : vector<64x64xf32> -> vector<4x4x4xf32>
  func.return %simt : vector<4x4x4xf32>
}
// CHECK-LABEL: func.func @to_simt_op
// CHECK:      iree_vector_ext.to_simd
