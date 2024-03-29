// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Ordering matters when multiple lines have the same types and tile shape and
// are supported by the CPU. In that case, the last-enumerated line overrides
// preceding lines. Always go from oldest to shiniest code path.
IREE_UK_MMT4D_TILE(x86_64, s8, s8, s32, 1, 8, 2, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, s8, s8, s32, 2, 8, 2, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, s8, s8, s32, 4, 8, 2, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, s8, s8, s32, 8, 8, 2, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, s16, s16, s32, 1, 8, 2, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, s16, s16, s32, 2, 8, 2, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, s16, s16, s32, 4, 8, 2, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, s16, s16, s32, 8, 8, 2, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, f32, f32, f32, 1, 8, 1, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, f32, f32, f32, 2, 8, 1, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, f32, f32, f32, 4, 8, 1, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, f32, f32, f32, 8, 8, 1, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, f16, f16, f32, 1, 8, 1, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, f16, f16, f32, 2, 8, 1, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, f16, f16, f32, 4, 8, 1, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, f16, f16, f32, 8, 8, 1, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, f16, f16, f16, 1, 8, 1, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, f16, f16, f16, 2, 8, 1, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, f16, f16, f16, 4, 8, 1, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, f16, f16, f16, 8, 8, 1, _avx2_fma)
IREE_UK_MMT4D_TILE(x86_64, f32, f32, f32, 1, 16, 1, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, f32, f32, f32, 2, 16, 1, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, f32, f32, f32, 4, 16, 1, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, f32, f32, f32, 8, 16, 1, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, f32, f32, f32, 16, 16, 1, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, f16, f16, f32, 1, 16, 1, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, f16, f16, f32, 2, 16, 1, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, f16, f16, f32, 4, 16, 1, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, f16, f16, f32, 8, 16, 1, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, f16, f16, f32, 16, 16, 1, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, bf16, bf16, f32, 1, 16, 2, _avx512_bf16)
IREE_UK_MMT4D_TILE(x86_64, bf16, bf16, f32, 2, 16, 2, _avx512_bf16)
IREE_UK_MMT4D_TILE(x86_64, bf16, bf16, f32, 4, 16, 2, _avx512_bf16)
IREE_UK_MMT4D_TILE(x86_64, bf16, bf16, f32, 8, 16, 2, _avx512_bf16)
IREE_UK_MMT4D_TILE(x86_64, bf16, bf16, f32, 16, 16, 2, _avx512_bf16)
IREE_UK_MMT4D_TILE(x86_64, f16, f16, f16, 1, 16, 1, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, f16, f16, f16, 2, 16, 1, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, f16, f16, f16, 4, 16, 1, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, f16, f16, f16, 8, 16, 1, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, f16, f16, f16, 16, 16, 1, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, bf16, bf16, bf16, 1, 16, 2, _avx512_bf16)
IREE_UK_MMT4D_TILE(x86_64, bf16, bf16, bf16, 2, 16, 2, _avx512_bf16)
IREE_UK_MMT4D_TILE(x86_64, bf16, bf16, bf16, 4, 16, 2, _avx512_bf16)
IREE_UK_MMT4D_TILE(x86_64, bf16, bf16, bf16, 8, 16, 2, _avx512_bf16)
IREE_UK_MMT4D_TILE(x86_64, bf16, bf16, bf16, 16, 16, 2, _avx512_bf16)
IREE_UK_MMT4D_TILE(x86_64, s8, s8, s32, 1, 16, 2, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, s8, s8, s32, 2, 16, 2, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, s8, s8, s32, 4, 16, 2, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, s8, s8, s32, 8, 16, 2, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, s8, s8, s32, 16, 16, 2, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, s8, s8, s32, 1, 16, 2, _avx512_vnni)
IREE_UK_MMT4D_TILE(x86_64, s8, s8, s32, 2, 16, 2, _avx512_vnni)
IREE_UK_MMT4D_TILE(x86_64, s8, s8, s32, 4, 16, 2, _avx512_vnni)
IREE_UK_MMT4D_TILE(x86_64, s8, s8, s32, 8, 16, 2, _avx512_vnni)
IREE_UK_MMT4D_TILE(x86_64, s8, s8, s32, 16, 16, 2, _avx512_vnni)
IREE_UK_MMT4D_TILE(x86_64, s16, s16, s32, 1, 16, 2, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, s16, s16, s32, 2, 16, 2, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, s16, s16, s32, 4, 16, 2, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, s16, s16, s32, 8, 16, 2, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, s16, s16, s32, 16, 16, 2, _avx512_base)
IREE_UK_MMT4D_TILE(x86_64, s16, s16, s32, 1, 16, 2, _avx512_vnni)
IREE_UK_MMT4D_TILE(x86_64, s16, s16, s32, 2, 16, 2, _avx512_vnni)
IREE_UK_MMT4D_TILE(x86_64, s16, s16, s32, 4, 16, 2, _avx512_vnni)
IREE_UK_MMT4D_TILE(x86_64, s16, s16, s32, 8, 16, 2, _avx512_vnni)
IREE_UK_MMT4D_TILE(x86_64, s16, s16, s32, 16, 16, 2, _avx512_vnni)
IREE_UK_MMT4D_TILE(x86_64, s16, u4, s32, 1, 32, 8, _avx512_vnni)
