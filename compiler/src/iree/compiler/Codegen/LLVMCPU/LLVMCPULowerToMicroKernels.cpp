// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/MicroKernelOps.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {
struct LLVMCPULowerToMicroKernelsPass
    : LLVMCPULowerToMicroKernelsBase<LLVMCPULowerToMicroKernelsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect>();
  }
  void runOnOperation() override;
};
}  // namespace

/// Returns `true` if an `outsOperand` value is initialized to zero.
static bool isInitializedToZero(Value outsOperand) {
  auto fillOp = outsOperand.getDefiningOp<linalg::FillOp>();
  if (!fillOp) return false;
  Value fillVal = fillOp.getDpsInputOperand(0)->get();
  return matchPattern(fillVal, m_Zero()) ||
         matchPattern(fillVal, m_AnyZeroFloat());
}

/// Matches an (linalg.fill -> )? linalg.mmt4d operation sequence and converts
/// it into a iree_codegen.mmt4d.ukernel operation, that is later lowered
/// into a call to the microkernel.
static FailureOr<IREE::Codegen::MicroKernelOpInterface> matchDAGForMicroKernel(
    RewriterBase &rewriter, linalg::Mmt4DOp op) {
  if (!op.hasTensorSemantics()) {
    return rewriter.notifyMatchFailure(
        op, "operations needs to have tensor semantics");
  }
  Value lhs = op.getDpsInputOperand(0)->get();
  Value rhs = op.getDpsInputOperand(1)->get();
  Value out = op.getDpsInitOperand(0)->get();
  auto lhsType = lhs.getType().cast<ShapedType>();
  auto rhsType = rhs.getType().cast<ShapedType>();
  auto outType = out.getType().cast<ShapedType>();
  Type lhsElemType = lhsType.getElementType();
  Type rhsElemType = rhsType.getElementType();
  Type outElemType = outType.getElementType();
  if (!((lhsElemType.isSignlessInteger(8) && rhsElemType.isSignlessInteger(8) &&
         outElemType.isSignlessInteger(32)) ||
        (lhsElemType.isF32() && rhsElemType.isF32() && outElemType.isF32()))) {
    return rewriter.notifyMatchFailure(
        op, "unsupported type for lowering to microkernels");
  }
  // check if the result has to be accumulated into a buffer.
  bool accumulate = !isInitializedToZero(out);
  if (!accumulate) {
    // Update the `out` value to encompass the dest of the op.
    if (auto fillOp = out.getDefiningOp<linalg::FillOp>()) {
      out = fillOp.getDpsInitOperand(0)->get();
    }
  }
  Location loc = op.getLoc();
  auto microKernelOp = rewriter.create<IREE::Codegen::Mmt4DUKernelOp>(
      loc, outType, lhs, rhs, out, rewriter.getBoolAttr(accumulate));
  return cast<IREE::Codegen::MicroKernelOpInterface>(
      microKernelOp.getOperation());
}

/// Matches an (linalg.fill -> )? linalg.matmul operation sequence and converts
/// it into a iree_codegen.generic.ukernel operation, that is lowered
/// into a call to the microkernel.
static FailureOr<IREE::Codegen::MicroKernelOpInterface> matchDAGForMicroKernel(
    RewriterBase &rewriter, linalg::MatmulOp op) {
  if (!op.hasTensorSemantics()) {
    return rewriter.notifyMatchFailure(
        op, "operations needs to have tensor semantics");
  }
  Value lhs = op.getDpsInputOperand(0)->get();
  Value rhs = op.getDpsInputOperand(1)->get();
  Value out = op.getDpsInitOperand(0)->get();
  auto lhsType = lhs.getType().cast<ShapedType>();
  auto rhsType = rhs.getType().cast<ShapedType>();
  auto outType = out.getType().cast<ShapedType>();
  std::string fnName = "";
  Type lhsElemType = lhsType.getElementType();
  Type rhsElemType = rhsType.getElementType();
  Type outElemType = outType.getElementType();
  if (lhsElemType.isSignlessInteger(8) && rhsElemType.isSignlessInteger(8) &&
      outElemType.isSignlessInteger(32)) {
    fnName = "vmvx.matmul.i8.i8.i32";
  } else if (lhsElemType.isF32() && rhsElemType.isF32() &&
             outElemType.isF32()) {
    fnName = "vmvx.matmul.f32.f32.f32";
  }
  if (fnName.empty()) {
    return rewriter.notifyMatchFailure(op,
                                       "unable to match micro kernel to op");
  }
  bool accumulate = !isInitializedToZero(out);
  int flags = 0;
  if (accumulate) {
    flags |= IREE_UK_FLAG_ACCUMULATE;
  } else {  // Update the `out` value to encompass the dest of the op.
    if (auto fillOp = out.getDefiningOp<linalg::FillOp>()) {
      out = fillOp.getDpsInitOperand(0)->get();
    }
  }

  Location loc = op.getLoc();
  Value m = rewriter.create<tensor::DimOp>(loc, lhs, 0);
  Value n = rewriter.create<tensor::DimOp>(loc, rhs, 1);
  Value k = rewriter.create<tensor::DimOp>(loc, out, 1);
  Value flagsVal = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(flags));
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::GenericUKernelOp>(
      loc, outType, fnName, ValueRange{lhs, rhs}, out,
      ValueRange{m, n, k, flagsVal});
  return cast<IREE::Codegen::MicroKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

namespace {

/// Pattern to lower (linalg.fill -> )? linalg.matmul operation sequence and
/// converts it into a iree_codegen.generic.ukernel operation
struct MatmulMicroKernelPattern : OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<IREE::Codegen::MicroKernelOpInterface> microKernelOp =
        matchDAGForMicroKernel(rewriter, matmulOp);
    if (failed(microKernelOp)) {
      return rewriter.notifyMatchFailure(
          matmulOp, "failed to find microkernel op to replace with");
    }
    rewriter.replaceOp(matmulOp, microKernelOp.value()->getResults());
    return success();
  }
};

/// Pattern to lower (linalg.fill -> )? linalg.mmt4d operation sequence and
/// converts it into a iree_codegen.mmt4d.ukernel operation
struct Mmt4DUKernelPattern : OpRewritePattern<linalg::Mmt4DOp> {
  using OpRewritePattern<linalg::Mmt4DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Mmt4DOp matmulOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<IREE::Codegen::MicroKernelOpInterface> microKernelOp =
        matchDAGForMicroKernel(rewriter, matmulOp);
    if (failed(microKernelOp)) {
      return rewriter.notifyMatchFailure(
          matmulOp, "failed to find microkernel op to replace with");
    }
    rewriter.replaceOp(matmulOp, microKernelOp.value()->getResults());
    return success();
  }
};
}  // namespace

void LLVMCPULowerToMicroKernelsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<MatmulMicroKernelPattern, Mmt4DUKernelPattern>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<>> createLLVMCPULowerToMicroKernelsPass() {
  return std::make_unique<LLVMCPULowerToMicroKernelsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
