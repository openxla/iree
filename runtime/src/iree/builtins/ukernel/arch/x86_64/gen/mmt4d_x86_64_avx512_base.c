// Auto-generated file. Do not edit!
//   Template:
//   ./runtime/src/iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64_avx512_base.c.in
//   Generator: ./third_party/XNNPACK/tools/xngen.py
//
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <immintrin.h>

#include "iree/builtins/ukernel/arch/x86_64/common_x86_64.h"
#include "iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64_internal.h"

void iree_uk_mmt4d_tile_f32f32f32_1x16x1_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  float* IREE_UK_RESTRICT out_ptr = out_tile;
  const float* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // The prefetches in this function are motivated by benchmarking on
  // Skylake; their effect was a > 1.3x speedup on 1024x1024 matmuls. The
  // prefetch-ahead offset of 128*sizeof(float) in the loop was empirically
  // determined. Similar prefetches did not produce any benefit in other
  // kernels, even though they are very similar to this one.
  _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
  _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
  __m512 acc0;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    acc0 = _mm512_loadu_ps(out_ptr + 0 * 16);
  } else {
    acc0 = _mm512_setzero_ps();
  }

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m512 rhs = _mm512_loadu_ps(rhs_ptr);
    _mm_prefetch((const char*)(rhs_ptr + 128), _MM_HINT_T0);
    rhs_ptr += 16;
    acc0 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[0]), rhs, acc0);
    _mm_prefetch((const char*)(lhs_ptr + 128), _MM_HINT_T0);
    lhs_ptr += 1;
  }

  _mm512_storeu_ps(out_ptr + 0 * 16, acc0);
}
void iree_uk_mmt4d_tile_f32f32f32_2x16x1_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  float* IREE_UK_RESTRICT out_ptr = out_tile;
  const float* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // The prefetches in this function are motivated by benchmarking on
  // Skylake; their effect was a > 1.3x speedup on 1024x1024 matmuls. The
  // prefetch-ahead offset of 128*sizeof(float) in the loop was empirically
  // determined. Similar prefetches did not produce any benefit in other
  // kernels, even though they are very similar to this one.
  _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
  _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
  __m512 acc0;
  __m512 acc1;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    acc0 = _mm512_loadu_ps(out_ptr + 0 * 16);
    acc1 = _mm512_loadu_ps(out_ptr + 1 * 16);
  } else {
    acc0 = _mm512_setzero_ps();
    acc1 = _mm512_setzero_ps();
  }

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m512 rhs = _mm512_loadu_ps(rhs_ptr);
    _mm_prefetch((const char*)(rhs_ptr + 128), _MM_HINT_T0);
    rhs_ptr += 16;
    acc0 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[0]), rhs, acc0);
    acc1 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[1]), rhs, acc1);
    _mm_prefetch((const char*)(lhs_ptr + 128), _MM_HINT_T0);
    lhs_ptr += 2;
  }

  _mm512_storeu_ps(out_ptr + 0 * 16, acc0);
  _mm512_storeu_ps(out_ptr + 1 * 16, acc1);
}
void iree_uk_mmt4d_tile_f32f32f32_4x16x1_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  float* IREE_UK_RESTRICT out_ptr = out_tile;
  const float* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // The prefetches in this function are motivated by benchmarking on
  // Skylake; their effect was a > 1.3x speedup on 1024x1024 matmuls. The
  // prefetch-ahead offset of 128*sizeof(float) in the loop was empirically
  // determined. Similar prefetches did not produce any benefit in other
  // kernels, even though they are very similar to this one.
  _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
  _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
  __m512 acc0;
  __m512 acc1;
  __m512 acc2;
  __m512 acc3;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    acc0 = _mm512_loadu_ps(out_ptr + 0 * 16);
    acc1 = _mm512_loadu_ps(out_ptr + 1 * 16);
    acc2 = _mm512_loadu_ps(out_ptr + 2 * 16);
    acc3 = _mm512_loadu_ps(out_ptr + 3 * 16);
  } else {
    acc0 = _mm512_setzero_ps();
    acc1 = _mm512_setzero_ps();
    acc2 = _mm512_setzero_ps();
    acc3 = _mm512_setzero_ps();
  }

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m512 rhs = _mm512_loadu_ps(rhs_ptr);
    _mm_prefetch((const char*)(rhs_ptr + 128), _MM_HINT_T0);
    rhs_ptr += 16;
    acc0 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[0]), rhs, acc0);
    acc1 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[1]), rhs, acc1);
    acc2 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[2]), rhs, acc2);
    acc3 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[3]), rhs, acc3);
    _mm_prefetch((const char*)(lhs_ptr + 128), _MM_HINT_T0);
    lhs_ptr += 4;
  }

  _mm512_storeu_ps(out_ptr + 0 * 16, acc0);
  _mm512_storeu_ps(out_ptr + 1 * 16, acc1);
  _mm512_storeu_ps(out_ptr + 2 * 16, acc2);
  _mm512_storeu_ps(out_ptr + 3 * 16, acc3);
}
void iree_uk_mmt4d_tile_f32f32f32_8x16x1_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  float* IREE_UK_RESTRICT out_ptr = out_tile;
  const float* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // The prefetches in this function are motivated by benchmarking on
  // Skylake; their effect was a > 1.3x speedup on 1024x1024 matmuls. The
  // prefetch-ahead offset of 128*sizeof(float) in the loop was empirically
  // determined. Similar prefetches did not produce any benefit in other
  // kernels, even though they are very similar to this one.
  _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
  _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
  __m512 acc0;
  __m512 acc1;
  __m512 acc2;
  __m512 acc3;
  __m512 acc4;
  __m512 acc5;
  __m512 acc6;
  __m512 acc7;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    acc0 = _mm512_loadu_ps(out_ptr + 0 * 16);
    acc1 = _mm512_loadu_ps(out_ptr + 1 * 16);
    acc2 = _mm512_loadu_ps(out_ptr + 2 * 16);
    acc3 = _mm512_loadu_ps(out_ptr + 3 * 16);
    acc4 = _mm512_loadu_ps(out_ptr + 4 * 16);
    acc5 = _mm512_loadu_ps(out_ptr + 5 * 16);
    acc6 = _mm512_loadu_ps(out_ptr + 6 * 16);
    acc7 = _mm512_loadu_ps(out_ptr + 7 * 16);
  } else {
    acc0 = _mm512_setzero_ps();
    acc1 = _mm512_setzero_ps();
    acc2 = _mm512_setzero_ps();
    acc3 = _mm512_setzero_ps();
    acc4 = _mm512_setzero_ps();
    acc5 = _mm512_setzero_ps();
    acc6 = _mm512_setzero_ps();
    acc7 = _mm512_setzero_ps();
  }

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m512 rhs = _mm512_loadu_ps(rhs_ptr);
    _mm_prefetch((const char*)(rhs_ptr + 128), _MM_HINT_T0);
    rhs_ptr += 16;
    acc0 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[0]), rhs, acc0);
    acc1 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[1]), rhs, acc1);
    acc2 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[2]), rhs, acc2);
    acc3 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[3]), rhs, acc3);
    acc4 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[4]), rhs, acc4);
    acc5 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[5]), rhs, acc5);
    acc6 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[6]), rhs, acc6);
    acc7 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[7]), rhs, acc7);
    _mm_prefetch((const char*)(lhs_ptr + 128), _MM_HINT_T0);
    lhs_ptr += 8;
  }

  _mm512_storeu_ps(out_ptr + 0 * 16, acc0);
  _mm512_storeu_ps(out_ptr + 1 * 16, acc1);
  _mm512_storeu_ps(out_ptr + 2 * 16, acc2);
  _mm512_storeu_ps(out_ptr + 3 * 16, acc3);
  _mm512_storeu_ps(out_ptr + 4 * 16, acc4);
  _mm512_storeu_ps(out_ptr + 5 * 16, acc5);
  _mm512_storeu_ps(out_ptr + 6 * 16, acc6);
  _mm512_storeu_ps(out_ptr + 7 * 16, acc7);
}
void iree_uk_mmt4d_tile_f32f32f32_16x16x1_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  float* IREE_UK_RESTRICT out_ptr = out_tile;
  const float* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // The prefetches in this function are motivated by benchmarking on
  // Skylake; their effect was a > 1.3x speedup on 1024x1024 matmuls. The
  // prefetch-ahead offset of 128*sizeof(float) in the loop was empirically
  // determined. Similar prefetches did not produce any benefit in other
  // kernels, even though they are very similar to this one.
  _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
  _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
  __m512 acc0;
  __m512 acc1;
  __m512 acc2;
  __m512 acc3;
  __m512 acc4;
  __m512 acc5;
  __m512 acc6;
  __m512 acc7;
  __m512 acc8;
  __m512 acc9;
  __m512 acc10;
  __m512 acc11;
  __m512 acc12;
  __m512 acc13;
  __m512 acc14;
  __m512 acc15;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    acc0 = _mm512_loadu_ps(out_ptr + 0 * 16);
    acc1 = _mm512_loadu_ps(out_ptr + 1 * 16);
    acc2 = _mm512_loadu_ps(out_ptr + 2 * 16);
    acc3 = _mm512_loadu_ps(out_ptr + 3 * 16);
    acc4 = _mm512_loadu_ps(out_ptr + 4 * 16);
    acc5 = _mm512_loadu_ps(out_ptr + 5 * 16);
    acc6 = _mm512_loadu_ps(out_ptr + 6 * 16);
    acc7 = _mm512_loadu_ps(out_ptr + 7 * 16);
    acc8 = _mm512_loadu_ps(out_ptr + 8 * 16);
    acc9 = _mm512_loadu_ps(out_ptr + 9 * 16);
    acc10 = _mm512_loadu_ps(out_ptr + 10 * 16);
    acc11 = _mm512_loadu_ps(out_ptr + 11 * 16);
    acc12 = _mm512_loadu_ps(out_ptr + 12 * 16);
    acc13 = _mm512_loadu_ps(out_ptr + 13 * 16);
    acc14 = _mm512_loadu_ps(out_ptr + 14 * 16);
    acc15 = _mm512_loadu_ps(out_ptr + 15 * 16);
  } else {
    acc0 = _mm512_setzero_ps();
    acc1 = _mm512_setzero_ps();
    acc2 = _mm512_setzero_ps();
    acc3 = _mm512_setzero_ps();
    acc4 = _mm512_setzero_ps();
    acc5 = _mm512_setzero_ps();
    acc6 = _mm512_setzero_ps();
    acc7 = _mm512_setzero_ps();
    acc8 = _mm512_setzero_ps();
    acc9 = _mm512_setzero_ps();
    acc10 = _mm512_setzero_ps();
    acc11 = _mm512_setzero_ps();
    acc12 = _mm512_setzero_ps();
    acc13 = _mm512_setzero_ps();
    acc14 = _mm512_setzero_ps();
    acc15 = _mm512_setzero_ps();
  }

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m512 rhs = _mm512_loadu_ps(rhs_ptr);
    _mm_prefetch((const char*)(rhs_ptr + 128), _MM_HINT_T0);
    rhs_ptr += 16;
    acc0 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[0]), rhs, acc0);
    acc1 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[1]), rhs, acc1);
    acc2 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[2]), rhs, acc2);
    acc3 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[3]), rhs, acc3);
    acc4 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[4]), rhs, acc4);
    acc5 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[5]), rhs, acc5);
    acc6 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[6]), rhs, acc6);
    acc7 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[7]), rhs, acc7);
    acc8 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[8]), rhs, acc8);
    acc9 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[9]), rhs, acc9);
    acc10 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[10]), rhs, acc10);
    acc11 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[11]), rhs, acc11);
    acc12 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[12]), rhs, acc12);
    acc13 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[13]), rhs, acc13);
    acc14 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[14]), rhs, acc14);
    acc15 = _mm512_fmadd_ps(_mm512_set1_ps(lhs_ptr[15]), rhs, acc15);
    _mm_prefetch((const char*)(lhs_ptr + 128), _MM_HINT_T0);
    lhs_ptr += 16;
  }

  _mm512_storeu_ps(out_ptr + 0 * 16, acc0);
  _mm512_storeu_ps(out_ptr + 1 * 16, acc1);
  _mm512_storeu_ps(out_ptr + 2 * 16, acc2);
  _mm512_storeu_ps(out_ptr + 3 * 16, acc3);
  _mm512_storeu_ps(out_ptr + 4 * 16, acc4);
  _mm512_storeu_ps(out_ptr + 5 * 16, acc5);
  _mm512_storeu_ps(out_ptr + 6 * 16, acc6);
  _mm512_storeu_ps(out_ptr + 7 * 16, acc7);
  _mm512_storeu_ps(out_ptr + 8 * 16, acc8);
  _mm512_storeu_ps(out_ptr + 9 * 16, acc9);
  _mm512_storeu_ps(out_ptr + 10 * 16, acc10);
  _mm512_storeu_ps(out_ptr + 11 * 16, acc11);
  _mm512_storeu_ps(out_ptr + 12 * 16, acc12);
  _mm512_storeu_ps(out_ptr + 13 * 16, acc13);
  _mm512_storeu_ps(out_ptr + 14 * 16, acc14);
  _mm512_storeu_ps(out_ptr + 15 * 16, acc15);
}

// Shared implementation for f16f16f16 and f16f16f32.
// In the f16f16f16 case, intermediate roundings are skipped. This function
// should only be used if IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS is set.
void iree_uk_mmt4d_tile_f16f16f16_1x16x1_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const iree_uk_uint16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_uint16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // The prefetches in this function are carried over from
  // iree_uk_mmt4d_tile_f32f32f32_16x16x1_x86_64_avx512_base.
  _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
  _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
  __m512 acc0;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    iree_uk_uint16_t* IREE_UK_RESTRICT out_ptr = out_tile;
    acc0 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 0 * 16)));
  } else {
    acc0 = _mm512_setzero_ps();
  }

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m512 rhs = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)rhs_ptr));
    _mm_prefetch((const char*)(rhs_ptr + 128), _MM_HINT_T0);
    rhs_ptr += 16;
    acc0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[0])), rhs,
                           acc0);
    _mm_prefetch((const char*)(lhs_ptr + 128), _MM_HINT_T0);
    lhs_ptr += 1;
  }
  iree_uk_uint16_t* IREE_UK_RESTRICT out_ptr = out_tile;
  _mm256_storeu_si256((__m256i*)(out_ptr + 0 * 16),
                      _mm512_cvtps_ph(acc0, _MM_FROUND_TO_NEAREST_INT));
}
void iree_uk_mmt4d_tile_f16f16f16_2x16x1_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const iree_uk_uint16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_uint16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // The prefetches in this function are carried over from
  // iree_uk_mmt4d_tile_f32f32f32_16x16x1_x86_64_avx512_base.
  _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
  _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
  __m512 acc0;
  __m512 acc1;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    iree_uk_uint16_t* IREE_UK_RESTRICT out_ptr = out_tile;
    acc0 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 0 * 16)));
    acc1 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 1 * 16)));
  } else {
    acc0 = _mm512_setzero_ps();
    acc1 = _mm512_setzero_ps();
  }

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m512 rhs = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)rhs_ptr));
    _mm_prefetch((const char*)(rhs_ptr + 128), _MM_HINT_T0);
    rhs_ptr += 16;
    acc0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[0])), rhs,
                           acc0);
    acc1 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[1])), rhs,
                           acc1);
    _mm_prefetch((const char*)(lhs_ptr + 128), _MM_HINT_T0);
    lhs_ptr += 2;
  }
  iree_uk_uint16_t* IREE_UK_RESTRICT out_ptr = out_tile;
  _mm256_storeu_si256((__m256i*)(out_ptr + 0 * 16),
                      _mm512_cvtps_ph(acc0, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 1 * 16),
                      _mm512_cvtps_ph(acc1, _MM_FROUND_TO_NEAREST_INT));
}
void iree_uk_mmt4d_tile_f16f16f16_4x16x1_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const iree_uk_uint16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_uint16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // The prefetches in this function are carried over from
  // iree_uk_mmt4d_tile_f32f32f32_16x16x1_x86_64_avx512_base.
  _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
  _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
  __m512 acc0;
  __m512 acc1;
  __m512 acc2;
  __m512 acc3;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    iree_uk_uint16_t* IREE_UK_RESTRICT out_ptr = out_tile;
    acc0 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 0 * 16)));
    acc1 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 1 * 16)));
    acc2 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 2 * 16)));
    acc3 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 3 * 16)));
  } else {
    acc0 = _mm512_setzero_ps();
    acc1 = _mm512_setzero_ps();
    acc2 = _mm512_setzero_ps();
    acc3 = _mm512_setzero_ps();
  }

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m512 rhs = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)rhs_ptr));
    _mm_prefetch((const char*)(rhs_ptr + 128), _MM_HINT_T0);
    rhs_ptr += 16;
    acc0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[0])), rhs,
                           acc0);
    acc1 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[1])), rhs,
                           acc1);
    acc2 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[2])), rhs,
                           acc2);
    acc3 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[3])), rhs,
                           acc3);
    _mm_prefetch((const char*)(lhs_ptr + 128), _MM_HINT_T0);
    lhs_ptr += 4;
  }
  iree_uk_uint16_t* IREE_UK_RESTRICT out_ptr = out_tile;
  _mm256_storeu_si256((__m256i*)(out_ptr + 0 * 16),
                      _mm512_cvtps_ph(acc0, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 1 * 16),
                      _mm512_cvtps_ph(acc1, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 2 * 16),
                      _mm512_cvtps_ph(acc2, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 3 * 16),
                      _mm512_cvtps_ph(acc3, _MM_FROUND_TO_NEAREST_INT));
}
void iree_uk_mmt4d_tile_f16f16f16_8x16x1_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const iree_uk_uint16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_uint16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // The prefetches in this function are carried over from
  // iree_uk_mmt4d_tile_f32f32f32_16x16x1_x86_64_avx512_base.
  _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
  _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
  __m512 acc0;
  __m512 acc1;
  __m512 acc2;
  __m512 acc3;
  __m512 acc4;
  __m512 acc5;
  __m512 acc6;
  __m512 acc7;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    iree_uk_uint16_t* IREE_UK_RESTRICT out_ptr = out_tile;
    acc0 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 0 * 16)));
    acc1 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 1 * 16)));
    acc2 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 2 * 16)));
    acc3 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 3 * 16)));
    acc4 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 4 * 16)));
    acc5 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 5 * 16)));
    acc6 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 6 * 16)));
    acc7 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 7 * 16)));
  } else {
    acc0 = _mm512_setzero_ps();
    acc1 = _mm512_setzero_ps();
    acc2 = _mm512_setzero_ps();
    acc3 = _mm512_setzero_ps();
    acc4 = _mm512_setzero_ps();
    acc5 = _mm512_setzero_ps();
    acc6 = _mm512_setzero_ps();
    acc7 = _mm512_setzero_ps();
  }

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m512 rhs = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)rhs_ptr));
    _mm_prefetch((const char*)(rhs_ptr + 128), _MM_HINT_T0);
    rhs_ptr += 16;
    acc0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[0])), rhs,
                           acc0);
    acc1 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[1])), rhs,
                           acc1);
    acc2 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[2])), rhs,
                           acc2);
    acc3 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[3])), rhs,
                           acc3);
    acc4 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[4])), rhs,
                           acc4);
    acc5 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[5])), rhs,
                           acc5);
    acc6 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[6])), rhs,
                           acc6);
    acc7 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[7])), rhs,
                           acc7);
    _mm_prefetch((const char*)(lhs_ptr + 128), _MM_HINT_T0);
    lhs_ptr += 8;
  }
  iree_uk_uint16_t* IREE_UK_RESTRICT out_ptr = out_tile;
  _mm256_storeu_si256((__m256i*)(out_ptr + 0 * 16),
                      _mm512_cvtps_ph(acc0, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 1 * 16),
                      _mm512_cvtps_ph(acc1, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 2 * 16),
                      _mm512_cvtps_ph(acc2, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 3 * 16),
                      _mm512_cvtps_ph(acc3, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 4 * 16),
                      _mm512_cvtps_ph(acc4, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 5 * 16),
                      _mm512_cvtps_ph(acc5, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 6 * 16),
                      _mm512_cvtps_ph(acc6, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 7 * 16),
                      _mm512_cvtps_ph(acc7, _MM_FROUND_TO_NEAREST_INT));
}
void iree_uk_mmt4d_tile_f16f16f16_16x16x1_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const iree_uk_uint16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_uint16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // The prefetches in this function are carried over from
  // iree_uk_mmt4d_tile_f32f32f32_16x16x1_x86_64_avx512_base.
  _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
  _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
  __m512 acc0;
  __m512 acc1;
  __m512 acc2;
  __m512 acc3;
  __m512 acc4;
  __m512 acc5;
  __m512 acc6;
  __m512 acc7;
  __m512 acc8;
  __m512 acc9;
  __m512 acc10;
  __m512 acc11;
  __m512 acc12;
  __m512 acc13;
  __m512 acc14;
  __m512 acc15;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    iree_uk_uint16_t* IREE_UK_RESTRICT out_ptr = out_tile;
    acc0 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 0 * 16)));
    acc1 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 1 * 16)));
    acc2 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 2 * 16)));
    acc3 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 3 * 16)));
    acc4 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 4 * 16)));
    acc5 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 5 * 16)));
    acc6 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 6 * 16)));
    acc7 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 7 * 16)));
    acc8 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 8 * 16)));
    acc9 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(out_ptr + 9 * 16)));
    acc10 = _mm512_cvtph_ps(
        _mm256_loadu_si256((const __m256i*)(out_ptr + 10 * 16)));
    acc11 = _mm512_cvtph_ps(
        _mm256_loadu_si256((const __m256i*)(out_ptr + 11 * 16)));
    acc12 = _mm512_cvtph_ps(
        _mm256_loadu_si256((const __m256i*)(out_ptr + 12 * 16)));
    acc13 = _mm512_cvtph_ps(
        _mm256_loadu_si256((const __m256i*)(out_ptr + 13 * 16)));
    acc14 = _mm512_cvtph_ps(
        _mm256_loadu_si256((const __m256i*)(out_ptr + 14 * 16)));
    acc15 = _mm512_cvtph_ps(
        _mm256_loadu_si256((const __m256i*)(out_ptr + 15 * 16)));
  } else {
    acc0 = _mm512_setzero_ps();
    acc1 = _mm512_setzero_ps();
    acc2 = _mm512_setzero_ps();
    acc3 = _mm512_setzero_ps();
    acc4 = _mm512_setzero_ps();
    acc5 = _mm512_setzero_ps();
    acc6 = _mm512_setzero_ps();
    acc7 = _mm512_setzero_ps();
    acc8 = _mm512_setzero_ps();
    acc9 = _mm512_setzero_ps();
    acc10 = _mm512_setzero_ps();
    acc11 = _mm512_setzero_ps();
    acc12 = _mm512_setzero_ps();
    acc13 = _mm512_setzero_ps();
    acc14 = _mm512_setzero_ps();
    acc15 = _mm512_setzero_ps();
  }

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m512 rhs = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)rhs_ptr));
    _mm_prefetch((const char*)(rhs_ptr + 128), _MM_HINT_T0);
    rhs_ptr += 16;
    acc0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[0])), rhs,
                           acc0);
    acc1 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[1])), rhs,
                           acc1);
    acc2 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[2])), rhs,
                           acc2);
    acc3 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[3])), rhs,
                           acc3);
    acc4 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[4])), rhs,
                           acc4);
    acc5 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[5])), rhs,
                           acc5);
    acc6 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[6])), rhs,
                           acc6);
    acc7 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[7])), rhs,
                           acc7);
    acc8 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[8])), rhs,
                           acc8);
    acc9 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[9])), rhs,
                           acc9);
    acc10 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[10])),
                            rhs, acc10);
    acc11 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[11])),
                            rhs, acc11);
    acc12 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[12])),
                            rhs, acc12);
    acc13 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[13])),
                            rhs, acc13);
    acc14 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[14])),
                            rhs, acc14);
    acc15 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[15])),
                            rhs, acc15);
    _mm_prefetch((const char*)(lhs_ptr + 128), _MM_HINT_T0);
    lhs_ptr += 16;
  }
  iree_uk_uint16_t* IREE_UK_RESTRICT out_ptr = out_tile;
  _mm256_storeu_si256((__m256i*)(out_ptr + 0 * 16),
                      _mm512_cvtps_ph(acc0, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 1 * 16),
                      _mm512_cvtps_ph(acc1, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 2 * 16),
                      _mm512_cvtps_ph(acc2, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 3 * 16),
                      _mm512_cvtps_ph(acc3, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 4 * 16),
                      _mm512_cvtps_ph(acc4, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 5 * 16),
                      _mm512_cvtps_ph(acc5, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 6 * 16),
                      _mm512_cvtps_ph(acc6, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 7 * 16),
                      _mm512_cvtps_ph(acc7, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 8 * 16),
                      _mm512_cvtps_ph(acc8, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 9 * 16),
                      _mm512_cvtps_ph(acc9, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 10 * 16),
                      _mm512_cvtps_ph(acc10, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 11 * 16),
                      _mm512_cvtps_ph(acc11, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 12 * 16),
                      _mm512_cvtps_ph(acc12, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 13 * 16),
                      _mm512_cvtps_ph(acc13, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 14 * 16),
                      _mm512_cvtps_ph(acc14, _MM_FROUND_TO_NEAREST_INT));
  _mm256_storeu_si256((__m256i*)(out_ptr + 15 * 16),
                      _mm512_cvtps_ph(acc15, _MM_FROUND_TO_NEAREST_INT));
}
void iree_uk_mmt4d_tile_f16f16f32_1x16x1_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const iree_uk_uint16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_uint16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // The prefetches in this function are carried over from
  // iree_uk_mmt4d_tile_f32f32f32_16x16x1_x86_64_avx512_base.
  _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
  _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
  __m512 acc0;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    float* IREE_UK_RESTRICT out_ptr = out_tile;
    acc0 = _mm512_loadu_ps(out_ptr + 0 * 16);
  } else {
    acc0 = _mm512_setzero_ps();
  }

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m512 rhs = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)rhs_ptr));
    _mm_prefetch((const char*)(rhs_ptr + 128), _MM_HINT_T0);
    rhs_ptr += 16;
    acc0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[0])), rhs,
                           acc0);
    _mm_prefetch((const char*)(lhs_ptr + 128), _MM_HINT_T0);
    lhs_ptr += 1;
  }
  float* IREE_UK_RESTRICT out_ptr = out_tile;
  _mm512_storeu_ps(out_ptr + 0 * 16, acc0);
}
void iree_uk_mmt4d_tile_f16f16f32_2x16x1_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const iree_uk_uint16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_uint16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // The prefetches in this function are carried over from
  // iree_uk_mmt4d_tile_f32f32f32_16x16x1_x86_64_avx512_base.
  _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
  _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
  __m512 acc0;
  __m512 acc1;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    float* IREE_UK_RESTRICT out_ptr = out_tile;
    acc0 = _mm512_loadu_ps(out_ptr + 0 * 16);
    acc1 = _mm512_loadu_ps(out_ptr + 1 * 16);
  } else {
    acc0 = _mm512_setzero_ps();
    acc1 = _mm512_setzero_ps();
  }

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m512 rhs = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)rhs_ptr));
    _mm_prefetch((const char*)(rhs_ptr + 128), _MM_HINT_T0);
    rhs_ptr += 16;
    acc0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[0])), rhs,
                           acc0);
    acc1 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[1])), rhs,
                           acc1);
    _mm_prefetch((const char*)(lhs_ptr + 128), _MM_HINT_T0);
    lhs_ptr += 2;
  }
  float* IREE_UK_RESTRICT out_ptr = out_tile;
  _mm512_storeu_ps(out_ptr + 0 * 16, acc0);
  _mm512_storeu_ps(out_ptr + 1 * 16, acc1);
}
void iree_uk_mmt4d_tile_f16f16f32_4x16x1_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const iree_uk_uint16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_uint16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // The prefetches in this function are carried over from
  // iree_uk_mmt4d_tile_f32f32f32_16x16x1_x86_64_avx512_base.
  _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
  _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
  __m512 acc0;
  __m512 acc1;
  __m512 acc2;
  __m512 acc3;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    float* IREE_UK_RESTRICT out_ptr = out_tile;
    acc0 = _mm512_loadu_ps(out_ptr + 0 * 16);
    acc1 = _mm512_loadu_ps(out_ptr + 1 * 16);
    acc2 = _mm512_loadu_ps(out_ptr + 2 * 16);
    acc3 = _mm512_loadu_ps(out_ptr + 3 * 16);
  } else {
    acc0 = _mm512_setzero_ps();
    acc1 = _mm512_setzero_ps();
    acc2 = _mm512_setzero_ps();
    acc3 = _mm512_setzero_ps();
  }

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m512 rhs = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)rhs_ptr));
    _mm_prefetch((const char*)(rhs_ptr + 128), _MM_HINT_T0);
    rhs_ptr += 16;
    acc0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[0])), rhs,
                           acc0);
    acc1 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[1])), rhs,
                           acc1);
    acc2 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[2])), rhs,
                           acc2);
    acc3 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[3])), rhs,
                           acc3);
    _mm_prefetch((const char*)(lhs_ptr + 128), _MM_HINT_T0);
    lhs_ptr += 4;
  }
  float* IREE_UK_RESTRICT out_ptr = out_tile;
  _mm512_storeu_ps(out_ptr + 0 * 16, acc0);
  _mm512_storeu_ps(out_ptr + 1 * 16, acc1);
  _mm512_storeu_ps(out_ptr + 2 * 16, acc2);
  _mm512_storeu_ps(out_ptr + 3 * 16, acc3);
}
void iree_uk_mmt4d_tile_f16f16f32_8x16x1_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const iree_uk_uint16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_uint16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // The prefetches in this function are carried over from
  // iree_uk_mmt4d_tile_f32f32f32_16x16x1_x86_64_avx512_base.
  _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
  _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
  __m512 acc0;
  __m512 acc1;
  __m512 acc2;
  __m512 acc3;
  __m512 acc4;
  __m512 acc5;
  __m512 acc6;
  __m512 acc7;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    float* IREE_UK_RESTRICT out_ptr = out_tile;
    acc0 = _mm512_loadu_ps(out_ptr + 0 * 16);
    acc1 = _mm512_loadu_ps(out_ptr + 1 * 16);
    acc2 = _mm512_loadu_ps(out_ptr + 2 * 16);
    acc3 = _mm512_loadu_ps(out_ptr + 3 * 16);
    acc4 = _mm512_loadu_ps(out_ptr + 4 * 16);
    acc5 = _mm512_loadu_ps(out_ptr + 5 * 16);
    acc6 = _mm512_loadu_ps(out_ptr + 6 * 16);
    acc7 = _mm512_loadu_ps(out_ptr + 7 * 16);
  } else {
    acc0 = _mm512_setzero_ps();
    acc1 = _mm512_setzero_ps();
    acc2 = _mm512_setzero_ps();
    acc3 = _mm512_setzero_ps();
    acc4 = _mm512_setzero_ps();
    acc5 = _mm512_setzero_ps();
    acc6 = _mm512_setzero_ps();
    acc7 = _mm512_setzero_ps();
  }

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m512 rhs = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)rhs_ptr));
    _mm_prefetch((const char*)(rhs_ptr + 128), _MM_HINT_T0);
    rhs_ptr += 16;
    acc0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[0])), rhs,
                           acc0);
    acc1 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[1])), rhs,
                           acc1);
    acc2 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[2])), rhs,
                           acc2);
    acc3 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[3])), rhs,
                           acc3);
    acc4 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[4])), rhs,
                           acc4);
    acc5 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[5])), rhs,
                           acc5);
    acc6 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[6])), rhs,
                           acc6);
    acc7 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[7])), rhs,
                           acc7);
    _mm_prefetch((const char*)(lhs_ptr + 128), _MM_HINT_T0);
    lhs_ptr += 8;
  }
  float* IREE_UK_RESTRICT out_ptr = out_tile;
  _mm512_storeu_ps(out_ptr + 0 * 16, acc0);
  _mm512_storeu_ps(out_ptr + 1 * 16, acc1);
  _mm512_storeu_ps(out_ptr + 2 * 16, acc2);
  _mm512_storeu_ps(out_ptr + 3 * 16, acc3);
  _mm512_storeu_ps(out_ptr + 4 * 16, acc4);
  _mm512_storeu_ps(out_ptr + 5 * 16, acc5);
  _mm512_storeu_ps(out_ptr + 6 * 16, acc6);
  _mm512_storeu_ps(out_ptr + 7 * 16, acc7);
}
void iree_uk_mmt4d_tile_f16f16f32_16x16x1_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const iree_uk_uint16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_uint16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // The prefetches in this function are carried over from
  // iree_uk_mmt4d_tile_f32f32f32_16x16x1_x86_64_avx512_base.
  _mm_prefetch((const char*)lhs_ptr, _MM_HINT_T0);
  _mm_prefetch((const char*)rhs_ptr, _MM_HINT_T0);
  __m512 acc0;
  __m512 acc1;
  __m512 acc2;
  __m512 acc3;
  __m512 acc4;
  __m512 acc5;
  __m512 acc6;
  __m512 acc7;
  __m512 acc8;
  __m512 acc9;
  __m512 acc10;
  __m512 acc11;
  __m512 acc12;
  __m512 acc13;
  __m512 acc14;
  __m512 acc15;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    float* IREE_UK_RESTRICT out_ptr = out_tile;
    acc0 = _mm512_loadu_ps(out_ptr + 0 * 16);
    acc1 = _mm512_loadu_ps(out_ptr + 1 * 16);
    acc2 = _mm512_loadu_ps(out_ptr + 2 * 16);
    acc3 = _mm512_loadu_ps(out_ptr + 3 * 16);
    acc4 = _mm512_loadu_ps(out_ptr + 4 * 16);
    acc5 = _mm512_loadu_ps(out_ptr + 5 * 16);
    acc6 = _mm512_loadu_ps(out_ptr + 6 * 16);
    acc7 = _mm512_loadu_ps(out_ptr + 7 * 16);
    acc8 = _mm512_loadu_ps(out_ptr + 8 * 16);
    acc9 = _mm512_loadu_ps(out_ptr + 9 * 16);
    acc10 = _mm512_loadu_ps(out_ptr + 10 * 16);
    acc11 = _mm512_loadu_ps(out_ptr + 11 * 16);
    acc12 = _mm512_loadu_ps(out_ptr + 12 * 16);
    acc13 = _mm512_loadu_ps(out_ptr + 13 * 16);
    acc14 = _mm512_loadu_ps(out_ptr + 14 * 16);
    acc15 = _mm512_loadu_ps(out_ptr + 15 * 16);
  } else {
    acc0 = _mm512_setzero_ps();
    acc1 = _mm512_setzero_ps();
    acc2 = _mm512_setzero_ps();
    acc3 = _mm512_setzero_ps();
    acc4 = _mm512_setzero_ps();
    acc5 = _mm512_setzero_ps();
    acc6 = _mm512_setzero_ps();
    acc7 = _mm512_setzero_ps();
    acc8 = _mm512_setzero_ps();
    acc9 = _mm512_setzero_ps();
    acc10 = _mm512_setzero_ps();
    acc11 = _mm512_setzero_ps();
    acc12 = _mm512_setzero_ps();
    acc13 = _mm512_setzero_ps();
    acc14 = _mm512_setzero_ps();
    acc15 = _mm512_setzero_ps();
  }

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m512 rhs = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)rhs_ptr));
    _mm_prefetch((const char*)(rhs_ptr + 128), _MM_HINT_T0);
    rhs_ptr += 16;
    acc0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[0])), rhs,
                           acc0);
    acc1 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[1])), rhs,
                           acc1);
    acc2 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[2])), rhs,
                           acc2);
    acc3 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[3])), rhs,
                           acc3);
    acc4 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[4])), rhs,
                           acc4);
    acc5 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[5])), rhs,
                           acc5);
    acc6 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[6])), rhs,
                           acc6);
    acc7 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[7])), rhs,
                           acc7);
    acc8 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[8])), rhs,
                           acc8);
    acc9 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[9])), rhs,
                           acc9);
    acc10 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[10])),
                            rhs, acc10);
    acc11 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[11])),
                            rhs, acc11);
    acc12 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[12])),
                            rhs, acc12);
    acc13 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[13])),
                            rhs, acc13);
    acc14 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[14])),
                            rhs, acc14);
    acc15 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_set1_epi16(lhs_ptr[15])),
                            rhs, acc15);
    _mm_prefetch((const char*)(lhs_ptr + 128), _MM_HINT_T0);
    lhs_ptr += 16;
  }
  float* IREE_UK_RESTRICT out_ptr = out_tile;
  _mm512_storeu_ps(out_ptr + 0 * 16, acc0);
  _mm512_storeu_ps(out_ptr + 1 * 16, acc1);
  _mm512_storeu_ps(out_ptr + 2 * 16, acc2);
  _mm512_storeu_ps(out_ptr + 3 * 16, acc3);
  _mm512_storeu_ps(out_ptr + 4 * 16, acc4);
  _mm512_storeu_ps(out_ptr + 5 * 16, acc5);
  _mm512_storeu_ps(out_ptr + 6 * 16, acc6);
  _mm512_storeu_ps(out_ptr + 7 * 16, acc7);
  _mm512_storeu_ps(out_ptr + 8 * 16, acc8);
  _mm512_storeu_ps(out_ptr + 9 * 16, acc9);
  _mm512_storeu_ps(out_ptr + 10 * 16, acc10);
  _mm512_storeu_ps(out_ptr + 11 * 16, acc11);
  _mm512_storeu_ps(out_ptr + 12 * 16, acc12);
  _mm512_storeu_ps(out_ptr + 13 * 16, acc13);
  _mm512_storeu_ps(out_ptr + 14 * 16, acc14);
  _mm512_storeu_ps(out_ptr + 15 * 16, acc15);
}

static inline void
iree_uk_mmt4d_tile_s8s8s32_1x16x2_to_16x16x2_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 16 && iree_uk_is_po2_u32(M0));
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // acc[i][0] contains the 1st 128bits of row i, the 2nd 128bits of row (i+4),
  //           the 3rd 128bits of row (i+8), the 4th 128bits of row (i+C).
  // The other acc[i][j] are permutations of these 128bits groups.
  // This unusual layout is chosen so that the inner arithmetic loop only needs
  // to perform cheap shuffles within 128bit groups of lanes.
  __m512i acc[4][4];
  const int imax = M0 <= 4 ? M0 : 4;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < imax; ++i) {
      for (int j = 0; j < 4; ++j) {
        if (M0 <= 8) {
          acc[i][j] = _mm512_castsi128_si512(
              _mm_loadu_si128((__m128i*)(out_ptr + i * 16 + j * 4)));
          if (M0 > 4) {
            acc[i][j] = _mm512_inserti32x4(
                acc[i][j],
                _mm_loadu_si128(
                    (__m128i*)(out_ptr + (i + 4) * 16 + ((5 - j) % 4) * 4)),
                1);
          }
        } else {
          acc[i][j] = iree_uk_avx512_loadu_4x128_from_16x16xi32(
              out_ptr, i, 4 * j, i + 4, 4 * ((5 - j) % 4), i + 8,
              4 * ((j + 2) % 4), i + 12, 4 * ((7 - j) % 4));
        }
      }
    }
  } else {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        acc[i][j] = _mm512_setzero_si512();
      }
    }
  }

  __m512i idx_45670123CDEF89AB =
      _mm512_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11);
  __m512i idx_89ABCDEF01234567 =
      _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
  __m512i idx_CDEF89AB45670123 =
      _mm512_setr_epi32(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m512i rhs_i16_perm[4];
    // rhs_i16_perm[0] is the rhs tile (2x8), sign-extended to i16.
    rhs_i16_perm[0] =
        _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)rhs_ptr));
    rhs_ptr += 32;
    // The other 3 rhs_i16_perm[i] are permutations of 128-bit groups of that.
    rhs_i16_perm[1] =
        _mm512_permutexvar_epi32(idx_45670123CDEF89AB, rhs_i16_perm[0]);
    rhs_i16_perm[2] =
        _mm512_permutexvar_epi32(idx_89ABCDEF01234567, rhs_i16_perm[0]);
    rhs_i16_perm[3] =
        _mm512_permutexvar_epi32(idx_CDEF89AB45670123, rhs_i16_perm[0]);
    // lhs_i16 is the lhs tile (M0x2), sign-extended to i16.
    __m512i lhs_i16;
    if (M0 == 1) {
      lhs_i16 =
          _mm512_castsi256_si512(_mm256_cvtepi8_epi16(_mm_loadu_si16(lhs_ptr)));
      lhs_ptr += 2;
    } else if (M0 == 2) {
      lhs_i16 =
          _mm512_castsi256_si512(_mm256_cvtepi8_epi16(_mm_loadu_si32(lhs_ptr)));
      lhs_ptr += 4;
    } else if (M0 == 4) {
      lhs_i16 =
          _mm512_castsi256_si512(_mm256_cvtepi8_epi16(_mm_loadu_si64(lhs_ptr)));
      lhs_ptr += 8;
    } else if (M0 == 8) {
      lhs_i16 = _mm512_castsi256_si512(
          _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)lhs_ptr)));
      lhs_ptr += 16;
    } else {
      lhs_i16 =
          _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)lhs_ptr));
      lhs_ptr += 32;
    }
    // lhs_i16_dup4[i] is lanes of lhs_i16 shuffled as:
    // (i, i, i, i, i+4, i+4, i+4, i+4, i+8, i+8, i+8, i+C, i+C, i+C, i+C).
    __m512i lhs_i16_dup4[4];
    if (M0 >= 1) lhs_i16_dup4[0] = _mm512_shuffle_epi32(lhs_i16, 0 * 0x55);
    if (M0 >= 2) lhs_i16_dup4[1] = _mm512_shuffle_epi32(lhs_i16, 1 * 0x55);
    if (M0 >= 4) lhs_i16_dup4[2] = _mm512_shuffle_epi32(lhs_i16, 2 * 0x55);
    if (M0 >= 4) lhs_i16_dup4[3] = _mm512_shuffle_epi32(lhs_i16, 3 * 0x55);
    for (int i = 0; i < imax; ++i) {
      for (int j = 0; j < 4; ++j) {
        acc[i][j] = _mm512_add_epi32(
            acc[i][j], _mm512_madd_epi16(lhs_i16_dup4[i], rhs_i16_perm[j]));
      }
    }
  }

  for (int i = 0; i < imax; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (M0 <= 8) {
        _mm_storeu_si128((__m128i*)(out_ptr + i * 16 + j * 4),
                         _mm512_extracti32x4_epi32(acc[i][j], 0));
        if (M0 > 4) {
          _mm_storeu_si128(
              (__m128i*)(out_ptr + (i + 4) * 16 + ((5 - j) % 4) * 4),
              _mm512_extracti32x4_epi32(acc[i][j], 1));
        }
      } else {
        iree_uk_avx512_storeu_4x128_to_16x16xi32(
            out_ptr, i, 4 * j, i + 4, 4 * ((5 - j) % 4), i + 8,
            4 * ((j + 2) % 4), i + 12, 4 * ((7 - j) % 4), acc[i][j]);
      }
    }
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8_16(
    iree_uk_mmt4d_tile_s8s8s32_1x16x2_to_16x16x2_x86_64_avx512_base,
    iree_uk_mmt4d_tile_s8s8s32_1x16x2_x86_64_avx512_base,
    iree_uk_mmt4d_tile_s8s8s32_2x16x2_x86_64_avx512_base,
    iree_uk_mmt4d_tile_s8s8s32_4x16x2_x86_64_avx512_base,
    iree_uk_mmt4d_tile_s8s8s32_8x16x2_x86_64_avx512_base,
    iree_uk_mmt4d_tile_s8s8s32_16x16x2_x86_64_avx512_base)

static inline void
iree_uk_mmt4d_tile_s16s16s32_1x16x2_to_16x16x2_x86_64_avx512_base(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 16 && iree_uk_is_po2_u32(M0));
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  // acc[i][0] contains the 1st 128bits of row i, the 2nd 128bits of row (i+4),
  //           the 3rd 128bits of row (i+8), the 4th 128bits of row (i+C).
  // The other acc[i][j] are permutations of these 128bits groups.
  // This unusual layout is chosen so that the inner arithmetic loop only needs
  // to perform cheap shuffles within 128bit groups of lanes.
  __m512i acc[4][4];
  const int imax = M0 <= 4 ? M0 : 4;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < imax; ++i) {
      for (int j = 0; j < 4; ++j) {
        if (M0 <= 8) {
          acc[i][j] = _mm512_castsi128_si512(
              _mm_loadu_si128((__m128i*)(out_ptr + i * 16 + j * 4)));
          if (M0 > 4) {
            acc[i][j] = _mm512_inserti32x4(
                acc[i][j],
                _mm_loadu_si128(
                    (__m128i*)(out_ptr + (i + 4) * 16 + ((5 - j) % 4) * 4)),
                1);
          }
        } else {
          acc[i][j] = iree_uk_avx512_loadu_4x128_from_16x16xi32(
              out_ptr, i, 4 * j, i + 4, 4 * ((5 - j) % 4), i + 8,
              4 * ((j + 2) % 4), i + 12, 4 * ((7 - j) % 4));
        }
      }
    }
  } else {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        acc[i][j] = _mm512_setzero_si512();
      }
    }
  }

  __m512i idx_45670123CDEF89AB =
      _mm512_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11);
  __m512i idx_89ABCDEF01234567 =
      _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
  __m512i idx_CDEF89AB45670123 =
      _mm512_setr_epi32(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);

  for (iree_uk_int32_t k = 0; k < params->K; ++k) {
    __m512i rhs_perm[4];
    // rhs_perm[0] is the rhs tile (2x8).
    rhs_perm[0] = _mm512_loadu_si512((const __m512i*)rhs_ptr);
    rhs_ptr += 32;
    // The other 3 rhs_perm[i] are permutations of 128-bit groups of that.
    rhs_perm[1] = _mm512_permutexvar_epi32(idx_45670123CDEF89AB, rhs_perm[0]);
    rhs_perm[2] = _mm512_permutexvar_epi32(idx_89ABCDEF01234567, rhs_perm[0]);
    rhs_perm[3] = _mm512_permutexvar_epi32(idx_CDEF89AB45670123, rhs_perm[0]);
    // lhs is the lhs tile (M0x2).
    __m512i lhs;
    if (M0 == 1) {
      lhs = _mm512_castsi128_si512(_mm_loadu_si32(lhs_ptr));
      lhs_ptr += 2;
    } else if (M0 == 2) {
      lhs = _mm512_castsi128_si512(_mm_loadu_si64(lhs_ptr));
      lhs_ptr += 4;
    } else if (M0 == 4) {
      lhs = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i*)lhs_ptr));
      lhs_ptr += 8;
    } else if (M0 == 8) {
      lhs = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)lhs_ptr));
      lhs_ptr += 16;
    } else {
      lhs = _mm512_loadu_si512((const __m512i*)lhs_ptr);
      lhs_ptr += 32;
    }
    // lhs_dup4[i] is lanes of lhs shuffled as:
    // (i, i, i, i, i+4, i+4, i+4, i+4, i+8, i+8, i+8, i+C, i+C, i+C, i+C).
    __m512i lhs_dup4[4];
    if (M0 >= 1) lhs_dup4[0] = _mm512_shuffle_epi32(lhs, 0 * 0x55);
    if (M0 >= 2) lhs_dup4[1] = _mm512_shuffle_epi32(lhs, 1 * 0x55);
    if (M0 >= 4) lhs_dup4[2] = _mm512_shuffle_epi32(lhs, 2 * 0x55);
    if (M0 >= 4) lhs_dup4[3] = _mm512_shuffle_epi32(lhs, 3 * 0x55);
    for (int i = 0; i < imax; ++i) {
      for (int j = 0; j < 4; ++j) {
        acc[i][j] = _mm512_add_epi32(
            acc[i][j], _mm512_madd_epi16(lhs_dup4[i], rhs_perm[j]));
      }
    }
  }

  for (int i = 0; i < imax; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (M0 <= 8) {
        _mm_storeu_si128((__m128i*)(out_ptr + i * 16 + j * 4),
                         _mm512_extracti32x4_epi32(acc[i][j], 0));
        if (M0 > 4) {
          _mm_storeu_si128(
              (__m128i*)(out_ptr + (i + 4) * 16 + ((5 - j) % 4) * 4),
              _mm512_extracti32x4_epi32(acc[i][j], 1));
        }
      } else {
        iree_uk_avx512_storeu_4x128_to_16x16xi32(
            out_ptr, i, 4 * j, i + 4, 4 * ((5 - j) % 4), i + 8,
            4 * ((j + 2) % 4), i + 12, 4 * ((7 - j) % 4), acc[i][j]);
      }
    }
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8_16(
    iree_uk_mmt4d_tile_s16s16s32_1x16x2_to_16x16x2_x86_64_avx512_base,
    iree_uk_mmt4d_tile_s16s16s32_1x16x2_x86_64_avx512_base,
    iree_uk_mmt4d_tile_s16s16s32_2x16x2_x86_64_avx512_base,
    iree_uk_mmt4d_tile_s16s16s32_4x16x2_x86_64_avx512_base,
    iree_uk_mmt4d_tile_s16s16s32_8x16x2_x86_64_avx512_base,
    iree_uk_mmt4d_tile_s16s16s32_16x16x2_x86_64_avx512_base)
