// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- PadTensorToInsertSlice.cpp ----- Pass to legalize linalg.pad_tensor-===//
//
// Pass to convert linalg.pad_tensor to linalg.fill + tensor.insert_slice
// operations which is the only way Vulkan backend can lower it to a single
// kernel.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {
/// Pattern to convert a linalg.pad_tensor operation into a fill + tensor
/// insert_slice. This is needed till pad_tensor op can be fused with its
/// consumers.
struct PadTensorOpConversion : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;
  PadTensorOpConversion(MLIRContext *context, bool skipOneLinalgUseCase)
      : OpRewritePattern<tensor::PadOp>(context, skipOneLinalgUseCase),
        skipOneLinalgUseCase(skipOneLinalgUseCase) {}

  LogicalResult matchAndRewrite(tensor::PadOp padTensorOp,
                                PatternRewriter &rewriter) const override {
    // Check that the region is just a yield operation which is returning a
    // scalar that is not one of the arguments of the linalg operation.
    Region &region = padTensorOp.getRegion();
    Block &block = region.front();
    if (!llvm::hasSingleElement(block)) return failure();
    auto yieldOp = cast<tensor::YieldOp>(block.getTerminator());
    Value yieldVal = yieldOp.getValue();
    if (llvm::any_of(block.getArguments(),
                     [&](Value v) { return v == yieldVal; })) {
      return failure();
    }

    if (skipOneLinalgUseCase && padTensorOp->hasOneUse()) {
      Operation *use = padTensorOp->use_begin()->getOwner();
      // TODO(#10312): Relax the condition to not check quantized ops. They
      // are going to be deprecated. We don't expect them being IREE's input.
      if (isa<linalg::LinalgOp>(use) &&
          !isa<linalg::Conv2DNhwcHwcfQOp, linalg::DepthwiseConv2DNhwcHwcQOp,
               linalg::DepthwiseConv2DNhwcHwcmQOp>(use)) {
        return failure();
      }
    }

    OpBuilder::InsertionGuard g(rewriter);
    Location loc = padTensorOp.getLoc();
    auto lowPad = padTensorOp.getMixedLowPad();
    auto highPad = padTensorOp.getMixedHighPad();
    Value source = padTensorOp.getSource();
    RankedTensorType sourceType = padTensorOp.getSourceType();
    int64_t rank = sourceType.getRank();

    // TODO(ravishankarm): Use shape inference interface to get this.
    SmallVector<OpFoldResult> sourceShape;
    SmallVector<OpFoldResult> outputShape;
    for (int64_t dim : llvm::seq<int64_t>(0, rank)) {
      SmallVector<Value> mapValues;
      Value sourceDim = rewriter.createOrFold<tensor::DimOp>(loc, source, dim);
      mapValues.push_back(sourceDim);
      if (auto cstDim = sourceDim.getDefiningOp<arith::ConstantIndexOp>()) {
        sourceShape.push_back(cstDim.getValue());
      } else {
        sourceShape.push_back(sourceDim);
      }
      AffineExpr expr = rewriter.getAffineDimExpr(0);
      unsigned numSymbols = 0;
      auto addValueOrAttr = [&](AffineExpr e, OpFoldResult valueOrAttr) {
        if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
          e = e + attr.cast<IntegerAttr>().getInt();
          return e;
        }
        e = e + rewriter.getAffineSymbolExpr(numSymbols++);
        mapValues.push_back(valueOrAttr.get<Value>());
        return e;
      };
      expr = addValueOrAttr(expr, lowPad[dim]);
      expr = addValueOrAttr(expr, highPad[dim]);
      Value v = applyMapToValues(
          rewriter, loc, AffineMap::get(1, numSymbols, expr), mapValues)[0];
      if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
        outputShape.push_back(cst.getValue());
      } else {
        outputShape.push_back(v);
      }
    }
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, outputShape, sourceType.getElementType());
    Value fill =
        rewriter.create<linalg::FillOp>(loc, yieldVal, initTensor).getResult(0);
    SmallVector<OpFoldResult> strides(rank, rewriter.getI64IntegerAttr(1));
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        padTensorOp, source, fill, lowPad, sourceShape, strides);
    return success();
  }

 private:
  // Option to skip the pattern when tensor.pad op has one use and is used by
  // a Linalg op.
  bool skipOneLinalgUseCase = false;
};

struct PadTensorToTensorInsertSlicePass
    : public PadTensorToTensorInsertSliceBase<
          PadTensorToTensorInsertSlicePass> {
  PadTensorToTensorInsertSlicePass(bool skipOneLinalgUseCase)
      : skipOneLinalgUseCase(skipOneLinalgUseCase) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, memref::MemRefDialect, func::FuncDialect,
                mlir::math::MathDialect, mlir::arith::ArithmeticDialect>();
  }

  LogicalResult initializeOptions(StringRef options) override {
    if (failed(Pass::initializeOptions(options))) {
      return failure();
    }
    // `skipOneLinalgUseCase` may have been set to `true` in the constructor
    // already. The |= is so we preserve that rather than overwrite it with the
    // default value `false` of `optionSkipOneLinalgUseCase`.
    skipOneLinalgUseCase |= optionSkipOneLinalgUseCase;
    return success();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<PadTensorOpConversion>(context, skipOneLinalgUseCase);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

 private:
  bool skipOneLinalgUseCase;
};

}  // namespace

std::unique_ptr<Pass> createPadTensorToTensorInsertSlicePass(
    bool skipOneLinalgUseCase) {
  return std::make_unique<PadTensorToTensorInsertSlicePass>(
      skipOneLinalgUseCase);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
