// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::GPU {

#define GEN_PASS_DEF_UNROLLTOINTRINSICSPASS
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h.inc"

namespace {
struct UnrollToIntrinsicsPass final
    : impl::UnrollToIntrinsicsPassBase<UnrollToIntrinsicsPass> {
  void runOnOperation() override;
};
} // namespace

static bool isReductionIterator(Attribute attr) {
  return cast<IREE::GPU::IteratorTypeAttr>(attr).getValue() ==
         utils::IteratorType::reduction;
}
static bool isParallelIterator(Attribute attr) {
  return cast<IREE::GPU::IteratorTypeAttr>(attr).getValue() ==
         utils::IteratorType::parallel;
}

/// Pick an unrolling order that reuses the LHS register.
static std::optional<SmallVector<int64_t>>
gpuMultiMmaUnrollOrder(Operation *op) {
  IREE::GPU::MultiMmaOp mmaOp = dyn_cast<IREE::GPU::MultiMmaOp>(op);
  if (!mmaOp) {
    return std::nullopt;
  }
  SmallVector<int64_t> order;
  // First make reduction the outer dimensions.
  for (auto [index, iter] : llvm::enumerate(mmaOp.getIteratorTypes())) {
    if (isReductionIterator(iter)) {
      order.push_back(index);
    }
  }

  llvm::SmallDenseSet<int64_t> dims;
  for (AffineExpr expr : mmaOp.getIndexingMapsArray()[0].getResults()) {
    dims.insert(cast<AffineDimExpr>(expr).getPosition());
  }
  // Then parallel dimensions that are part of Lhs as we want to re-use Lhs.
  for (auto [index, iter] : llvm::enumerate(mmaOp.getIteratorTypes())) {
    if (isParallelIterator(iter) && dims.count(index)) {
      order.push_back(index);
    }
  }
  // Then the remaining parallel loops.
  for (auto [index, iter] : llvm::enumerate(mmaOp.getIteratorTypes())) {
    if (isParallelIterator(iter) && !dims.count(index)) {
      order.push_back(index);
    }
  }
  return order;
}

static std::optional<SmallVector<int64_t>> getMultiMmaUnitShape(Operation *op) {
  IREE::GPU::MultiMmaOp mmaOp = dyn_cast<IREE::GPU::MultiMmaOp>(op);
  if (!mmaOp) {
    return std::nullopt;
  }
  SmallVector<int64_t> targetOuterShape(mmaOp.getIteratorTypes().size(), 1);
  return targetOuterShape;
}

void UnrollToIntrinsicsPass::runOnOperation() {
  MLIRContext *context = &getContext();

  {
    RewritePatternSet patterns(context);
    GPU::populateIREEGPUVectorUnrollPatterns(
        patterns, vector::UnrollVectorOptions()
                      .setNativeShapeFn(getMultiMmaUnitShape)
                      .setUnrollTraversalOrderFn(gpuMultiMmaUnrollOrder));
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  // Post unrolling unit dim folding patterns in preparation for later
  // lowerings.
  {
    RewritePatternSet patterns(context);
    GPU::populateIREEGPUDropUnitDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler::IREE::GPU
