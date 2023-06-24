// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Transforms.h - Transformations common to all backends --------------===//
//
// Defines transformations that are common to backends
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_CODEGEN_TRANSFORMS_TRANSFORMS_H_
#define IREE_COMPILER_CODEGEN_TRANSFORMS_TRANSFORMS_H_

#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

/// Get the `offsets`, `sizes` and `strides` for a `storeOp` (or `loadOp`). This
/// method clones the operations that generate the `Value`s used for
/// specifying the offsets, sizesm strides and dynamic dims of the
/// `storeOp/loadOp` at the insertion point to avoid use-def violations.
struct SliceAndDynamicDims {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  SmallVector<Value> dynamicDims;
};
SliceAndDynamicDims
cloneOffsetsSizesAndStrides(OpBuilder &builder,
                            IREE::Flow::DispatchTensorStoreOp storeOp);
SliceAndDynamicDims
cloneOffsetsSizesAndStrides(OpBuilder &builder,
                            IREE::Flow::DispatchTensorLoadOp loadOp);

/// Creates an allocation in the entry block of the function if the size is
/// statically bounded. For a static allocation, it returns an allocation
/// of the same size but in the entry basic block. For dynamic (still bounded)
/// allocations creates an allocation, and inserts a subview to match the
/// dynamic shape of the allocation. Returns std::nullopt if the method
/// couldnt creat an allocation in the entry block.
template <typename AllocLikeOpType>
std::optional<Value>
hoistOneStaticallyBoundAllocation(func::FuncOp funcOp, OpBuilder &builder,
                                  Location loc, MemRefType allocaType,
                                  ValueRange dynamicSizes,
                                  std::optional<uint64_t> alignment);

/// Hoists `allocaOp` to the entry block of the function if the size is
/// statically bounded. For a static allocation, it returns an allocation
/// of the same size but in the entry basic block. For dynamic (still bounded)
/// allocations creates an allocation, and inserts a subview to match the
/// dynamic shape of the allocation. The method returns a value, but
/// does not replace the uses of the `allocaOp`.
template <typename AllocLikeOpType>
std::optional<Value>
hoistOneStaticallyBoundAllocation(func::FuncOp funcOp, OpBuilder &builder,
                                  AllocLikeOpType allocaOp);

/// Traverse funcOp and try to hoist every AllocaOp to the entry block of the
/// function if the size is statically bounded.
template <typename AllocLikeOpType>
void hoistStaticallyBoundAllocationsInFunc(RewriterBase &rewriter,
                                           func::FuncOp funcOp);

/// Insert patterns to perform folding of AffineMinOp by matching the
/// pattern generated by tile and distribute. Try to fold a affine.min op by
/// matching the following form:
/// ```
/// scf.for %iv = %lb to %ub step %step
///   %affine.min affine_map<(d0, d1) -> (N, d0 - d1)>(%ub, %iv)
/// ```
/// With N a compile time constant. This operations can be replace by
/// `%cN = arith.constant N : index` if we can prove that %lb, %step and %ub
/// are divisible by N.
void populateAffineMinSCFCanonicalizationPattern(RewritePatternSet &patterns);

using GetMinMaxExprFn =
    std::function<std::optional<std::pair<AffineExpr, AffineExpr>>(
        Value value, SmallVectorImpl<Value> &dims,
        SmallVectorImpl<Value> &symbols)>;

/// Insert pattern to remove single iteration loop. The pattern will detect
/// single iteration loops based on the range returned by the lambda
/// |getMinMaxFn| for some know values.
void populateRemoveSingleIterationLoopPattern(RewritePatternSet &patterns,
                                              GetMinMaxExprFn getMinMaxFn);

/// Populate patterns that fold tensor.expand/collapse_shape into the source
/// hal.interface.binding.subspan.
void populateReshapeToInterfaceTensorPatterns(RewritePatternSet &patterns);

/// Populate patterns that remove dead allocations
void populateRemoveDeadMemAllocPatterns(RewritePatternSet &patterns);

// Group of Alloc operations that have overlapping liveranges.
using AliasGroup = SmallVector<Operation *>;

/// Analyze the liverange of the given allocs and set them in individual groups
/// if they don't overlap.
/// The algorithm is a simplistic memory allocation solution. It sorts
/// allocations into alias groups. Everytime two alloc's liverange interfers
/// they are merge into the same group. If a new alloc is part of multiple alias
/// groups all those are merged into one. At the end we are left with groups of
/// allocations that are disjoint and can use the same memory.
void analyseAllocsForPacking(func::FuncOp funcOp, ArrayRef<Operation *> allocs,
                             SmallVector<AliasGroup> &aliasGroups);

/// Pack groups of allocations into a unique large i8 allocation and use
/// memref.view to separate the indivudual allocations. This allows re-using
/// memory across alias groups.
void packAllocs(OpBuilder &builder, func::FuncOp funcOp,
                ArrayRef<AliasGroup> aliasGroups);

/// Lower the workgroup count region for the default code-generation path in
/// IREE. Given the list `workgroupCount` (fastest varying dimension innermost)
/// as computed within the `entryPointFn`, clones a backward slice of the
/// computation starting at these values and ending with
/// `flow.dispatch.constant_ordinal` into the workgroup count region on the
/// `hal.executable.export` op corresponding to the `entryPointFn`. Also removes
/// the `flow.dispatch.constant_ordinal` operations from within the
/// `entryPointFn`. Expects the workgroup count region of the corresponding
/// `hal.executable.export` to contain the
/// `flow.dispatch.workgroup_count_default` operation as a placeholder for the
/// computation to compute the number of workgroups. In absence of this
/// operation, this method does nothing assuming that the workgroup count
/// computation has already been resolved.
LogicalResult lowerWorkgroupCountFromSliceOp(
    RewriterBase &rewriter,
    IREE::Flow::DispatchWorkgroupCountFromSliceOp workgroupCountOp,
    func::FuncOp entryPointFn, ArrayRef<OpFoldResult> workgroupCount,
    int maxWorkgroupParallelDims = kNumMaxParallelDims);

/// Wrapper around `lowerWorkgroupCountFromSliceOp` method that
/// takes the `flow.dispatch.workgroup_count_from_slice` op
/// as an argument. Looks up the `hal.executable.export` operation
/// and finds the `flow.dispatch.workgroup_count_from_slice` op to lower.
LogicalResult lowerWorkgroupCountFromSliceOp(
    RewriterBase &rewriter, func::FuncOp entryPointFn,
    ArrayRef<OpFoldResult> workgroupCount,
    int maxWorkgroupParallelDims = kNumMaxParallelDims);

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_TRANSFORMS_TRANSFORMS_H_
