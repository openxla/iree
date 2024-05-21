// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

namespace {

// For narrowable inputs, selects
struct DemoteContractionInputsToBF16Pattern
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isaContractionOpInterface(linalgOp) &&
        !isa<linalg::ConvolutionOpInterface>(linalgOp.getOperation())) {
      return failure();
    }
    Type F32Type = rewriter.getF32Type();
    for (auto operand : linalgOp->getOperands()) {
      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandType || operandType.getElementType() != F32Type) {
        return failure();
      }
    }
    Location loc = linalgOp.getLoc();
    SmallVector<Value> demotedInputs;
    for (auto inputOperand : linalgOp.getDpsInputOperands()) {
      auto input = inputOperand->get();
      auto inputType = cast<RankedTensorType>(input.getType());
      auto demotedInputType =
          RankedTensorType::get(inputType.getShape(), rewriter.getBF16Type(),
                                inputType.getEncoding());
      SmallVector<AffineMap> maps(
          2, rewriter.getMultiDimIdentityMap(inputType.getRank()));
      SmallVector<utils::IteratorType> iteratorTypes(
          inputType.getRank(), utils::IteratorType::parallel);
      SmallVector<OpFoldResult> mixedSizes =
          tensor::getMixedSizes(rewriter, loc, input);
      Value empty = rewriter.create<tensor::EmptyOp>(loc, mixedSizes,
                                                     rewriter.getBF16Type());
      demotedInputs.push_back(
          rewriter
              .create<linalg::GenericOp>(
                  loc, TypeRange{demotedInputType}, ValueRange{input},
                  ValueRange{empty}, maps, iteratorTypes,
                  [&](OpBuilder &b, Location loc, ValueRange args) {
                    Value result = b.create<arith::TruncFOp>(
                        loc, rewriter.getBF16Type(), args[0]);
                    b.create<linalg::YieldOp>(loc, result);
                  })
              ->getResults()[0]);
    }

    if (auto genericOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation())) {
      rewriter.replaceOpWithNewOp<linalg::GenericOp>(
          linalgOp, linalgOp->getResultTypes(),
          /*inputs=*/demotedInputs, /*outputs=*/linalgOp.getDpsInits(),
          linalgOp.getIndexingMapsArray(), linalgOp.getIteratorTypesArray(),
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
            Value lhsPromoted = nestedBuilder.create<arith::ExtFOp>(
                nestedLoc, F32Type, args[0]);
            Value rhsPromoted = nestedBuilder.create<arith::ExtFOp>(
                nestedLoc, F32Type, args[1]);
            Value mul = nestedBuilder.create<arith::MulFOp>(
                nestedLoc, lhsPromoted, rhsPromoted);
            Value add =
                nestedBuilder.create<arith::AddFOp>(nestedLoc, mul, args[2]);
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
          },
          linalg::getPrunedAttributeList(genericOp));
      return success();
    }

    auto replaceOpInputs = [&](auto *typePtr) {
      auto namedOp = cast<std::remove_pointer_t<decltype(typePtr)>>(linalgOp);
      rewriter.replaceOpWithNewOp<std::remove_pointer_t<decltype(typePtr)>>(
          linalgOp, demotedInputs, linalgOp.getDpsInits(),
          linalg::getPrunedAttributeList(namedOp));
    };

    if (isa<linalg::MatmulOp>(linalgOp)) {
      replaceOpInputs(static_cast<linalg::MatmulOp *>(nullptr));
    } else if (isa<linalg::MatvecOp>(linalgOp)) {
      replaceOpInputs(static_cast<linalg::MatvecOp *>(nullptr));
    } else if (isa<linalg::VecmatOp>(linalgOp)) {
      replaceOpInputs(static_cast<linalg::VecmatOp *>(nullptr));
    } else if (isa<linalg::BatchMatmulOp>(linalgOp)) {
      replaceOpInputs(static_cast<linalg::BatchMatmulOp *>(nullptr));
    } else if (isa<linalg::BatchMatvecOp>(linalgOp)) {
      replaceOpInputs(static_cast<linalg::BatchMatvecOp *>(nullptr));
    } else if (isa<linalg::BatchVecmatOp>(linalgOp)) {
      replaceOpInputs(static_cast<linalg::BatchVecmatOp *>(nullptr));
    } else if (isa<linalg::MatmulTransposeAOp>(linalgOp)) {
      replaceOpInputs(static_cast<linalg::MatmulTransposeAOp *>(nullptr));
    } else if (isa<linalg::MatmulTransposeBOp>(linalgOp)) {
      replaceOpInputs(static_cast<linalg::MatmulTransposeBOp *>(nullptr));
    } else if (isa<linalg::BatchMatmulTransposeAOp>(linalgOp)) {
      replaceOpInputs(static_cast<linalg::BatchMatmulTransposeAOp *>(nullptr));
    } else if (isa<linalg::BatchMatmulTransposeBOp>(linalgOp)) {
      replaceOpInputs(static_cast<linalg::BatchMatmulTransposeBOp *>(nullptr));
    } else if (isa<linalg::Conv2DOp>(linalgOp)) {
      replaceOpInputs(static_cast<linalg::Conv2DOp *>(nullptr));
    } else if (isa<linalg::Conv2DNchwFchwOp>(linalgOp)) {
      replaceOpInputs(static_cast<linalg::Conv2DNchwFchwOp *>(nullptr));
    } else if (isa<linalg::Conv2DNhwcHwcfOp>(linalgOp)) {
      replaceOpInputs(static_cast<linalg::Conv2DNhwcHwcfOp *>(nullptr));
    } else if (isa<linalg::Conv2DNhwcFhwcOp>(linalgOp)) {
      replaceOpInputs(static_cast<linalg::Conv2DNhwcFhwcOp *>(nullptr));
    } else if (isa<linalg::Conv2DNgchwFgchwOp>(linalgOp)) {
      replaceOpInputs(static_cast<linalg::Conv2DNgchwFgchwOp *>(nullptr));
    } else if (isa<linalg::Conv2DNgchwGfchwOp>(linalgOp)) {
      replaceOpInputs(static_cast<linalg::Conv2DNgchwGfchwOp *>(nullptr));
    } else {
      return failure();
    }

    return success();
  }
};

class DemoteContractionInputsToBF16Pass
    : public DemoteContractionInputsToBF16Base<
          DemoteContractionInputsToBF16Pass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<DemoteContractionInputsToBF16Pattern>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createDemoteContractionInputsToBF16Pass() {
  return std::make_unique<DemoteContractionInputsToBF16Pass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
