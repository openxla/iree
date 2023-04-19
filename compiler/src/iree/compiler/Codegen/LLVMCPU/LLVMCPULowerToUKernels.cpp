// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/UKernelOps.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/EncodingInfo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {
struct LLVMCPULowerToUKernelsPass
    : LLVMCPULowerToUKernelsBase<LLVMCPULowerToUKernelsPass> {
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
/// it into a iree_codegen.ukernel.mmt4d operation, that is later lowered
/// into a call to the microkernel.
static FailureOr<IREE::Codegen::UKernelOpInterface> matchDAGForUKernel(
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
  std::string fnName;
  if (lhsElemType.isSignlessInteger(8) && rhsElemType.isSignlessInteger(8) &&
      outElemType.isSignlessInteger(32)) {
    fnName = "vmvx.mmt4d.i8i8i32";
  } else if (lhsElemType.isF32() && rhsElemType.isF32() &&
             outElemType.isF32()) {
    fnName = "vmvx.mmt4d.f32f32f32";
  }
  if (fnName.empty()) {
    return rewriter.notifyMatchFailure(
        op, "unsupported combination of element types");
  }

  // Check if the accumulator is zero-filled.
  int flags = 0;
  if (isInitializedToZero(out)) {
    // Not setting flags |= IREE_UK_FLAG_ACCUMULATE, so the mmt4d op won't read
    // the existing accumulator, so its defining op can be discarded.
    if (auto fillOp = out.getDefiningOp<linalg::FillOp>()) {
      out = fillOp.getDpsInitOperand(0)->get();
    }
  } else {
    // Tell the mmt4d op to read the existing accumulator.
    flags |= IREE_UK_FLAG_ACCUMULATE;
  }
  Location loc = op.getLoc();
  Value m = rewriter.create<tensor::DimOp>(loc, lhs, 0);
  Value n = rewriter.create<tensor::DimOp>(loc, rhs, 0);
  Value k = rewriter.create<tensor::DimOp>(loc, rhs, 1);
  Value m0 = rewriter.create<tensor::DimOp>(loc, lhs, 2);
  Value n0 = rewriter.create<tensor::DimOp>(loc, rhs, 2);
  Value k0 = rewriter.create<tensor::DimOp>(loc, rhs, 3);

  Value flagsVal = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(flags));
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fnName, ValueRange{lhs, rhs}, out,
      ValueRange{m, n, k, m0, n0, k0, flagsVal},
      /*strided_outer_dims=*/rewriter.getIndexAttr(1));
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

/// Matches an (linalg.fill -> )? linalg.matmul operation sequence and converts
/// it into a iree_codegen.ukernel.generic operation, that is lowered
/// into a call to the microkernel.
static FailureOr<IREE::Codegen::UKernelOpInterface> matchDAGForUKernel(
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
    fnName = "vmvx.matmul.i8i8i32";
  } else if (lhsElemType.isF32() && rhsElemType.isF32() &&
             outElemType.isF32()) {
    fnName = "vmvx.matmul.f32f32f32";
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
  Value k = rewriter.create<tensor::DimOp>(loc, lhs, 1);
  Value flagsVal = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(flags));
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fnName, ValueRange{lhs, rhs}, out,
      ValueRange{m, n, k, flagsVal},
      /*strided_outer_dims=*/rewriter.getIndexAttr(1));
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

static FailureOr<IREE::Codegen::UKernelOpInterface> matchDAGForUKernel(
    RewriterBase &rewriter, tensor::PackOp op) {
  if (!op.hasTensorSemantics()) {
    return rewriter.notifyMatchFailure(
        op, "operations needs to have tensor semantics");
  }
  Value in = op.getSource();
  Value out = op.getDest();
  auto inType = in.getType().cast<ShapedType>();
  auto outType = out.getType().cast<ShapedType>();
  Type inElemType = inType.getElementType();
  Type outElemType = outType.getElementType();
  std::string fnName;
  if (inElemType.isSignlessInteger(8) && outElemType.isSignlessInteger(8)) {
    fnName = "vmvx.pack.i8i8";
  } else if (inElemType.isSignlessInteger(32) &&
             outElemType.isSignlessInteger(32)) {
    fnName = "vmvx.pack.i32i32";
  } else if (inElemType.isF32() && outElemType.isF32()) {
    fnName = "vmvx.pack.f32f32";
  } else {
    return rewriter.notifyMatchFailure(
        op, "unsupported combination of element types");
  }

  if (inType.getRank() != 2) {
    return rewriter.notifyMatchFailure(op, "expected input to be 2D");
  }

  if (outType.getRank() != 4) {
    return rewriter.notifyMatchFailure(op, "expected output to be 4D");
  }

  int64_t innerDimsPos[2] = {0, 1};
  ArrayRef<int64_t> innerDimsPosArr = op.getInnerDimsPos();
  if (!innerDimsPosArr.empty()) {
    innerDimsPos[0] = innerDimsPosArr[0];
    innerDimsPos[1] = innerDimsPosArr[1];
  }

  int64_t outerDimsPerm[2] = {0, 1};
  ArrayRef<int64_t> outerDimsPosArr = op.getOuterDimsPerm();
  if (!outerDimsPosArr.empty()) {
    outerDimsPerm[0] = outerDimsPosArr[0];
    outerDimsPerm[1] = outerDimsPosArr[1];
  }

  int flags = 0;

  if (innerDimsPos[0] == 0 && innerDimsPos[1] == 1) {
    // nothing to do
  } else if (innerDimsPos[0] == 1 && innerDimsPos[1] == 0) {
    flags |= IREE_UK_FLAG_PACK_TRANSPOSE_INNER;
  } else {
    return rewriter.notifyMatchFailure(op, "unsupported inner_dims_pos");
  }

  if (outerDimsPerm[0] == 0 && outerDimsPerm[1] == 1) {
    // nothing to do
  } else if (outerDimsPerm[0] == 1 && outerDimsPerm[1] == 0) {
    flags |= IREE_UK_FLAG_PACK_TRANSPOSE_OUTER;
  } else {
    return rewriter.notifyMatchFailure(op, "unsupported outer_dims_perm");
  }

  Location loc = op.getLoc();
  Value paddingVal = op.getPaddingValue();
  if (!paddingVal) {
    paddingVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(inType.getElementType()),
        inType.getElementType());
  }

  Value in_size0 = rewriter.create<tensor::DimOp>(loc, in, 0);
  Value in_size1 = rewriter.create<tensor::DimOp>(loc, in, 1);
  Value out_size0 = rewriter.create<tensor::DimOp>(loc, out, 0);
  Value out_size1 = rewriter.create<tensor::DimOp>(loc, out, 1);
  Value out_size2 = rewriter.create<tensor::DimOp>(loc, out, 2);
  Value out_size3 = rewriter.create<tensor::DimOp>(loc, out, 3);
  Value flagsVal = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(flags));
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fnName, in, out,
      ValueRange{in_size0, in_size1, out_size0, out_size1, out_size2, out_size3,
                 paddingVal, flagsVal},
      /*strided_outer_dims=*/rewriter.getIndexAttr(1));
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

static FailureOr<IREE::Codegen::UKernelOpInterface> matchDAGForUKernel(
    RewriterBase &rewriter, tensor::UnPackOp op) {
  if (!op.hasTensorSemantics()) {
    return rewriter.notifyMatchFailure(
        op, "operations needs to have tensor semantics");
  }
  Value in = op.getSource();
  Value out = op.getDest();
  auto inType = in.getType().cast<ShapedType>();
  auto outType = out.getType().cast<ShapedType>();
  Type inElemType = inType.getElementType();
  Type outElemType = outType.getElementType();
  std::string fnName;
  if (inElemType.isSignlessInteger(32) && outElemType.isSignlessInteger(32)) {
    fnName = "vmvx.unpack.i32i32";
  } else if (inElemType.isF32() && outElemType.isF32()) {
    fnName = "vmvx.unpack.f32f32";
  } else {
    return rewriter.notifyMatchFailure(
        op, "unsupported combination of element types");
  }

  if (inType.getRank() != 4) {
    return rewriter.notifyMatchFailure(op, "expected input to be 4D");
  }

  if (outType.getRank() != 2) {
    return rewriter.notifyMatchFailure(op, "expected output to be 2D");
  }

  int64_t innerDimsPos[2] = {0, 1};
  ArrayRef<int64_t> innerDimsPosArr = op.getInnerDimsPos();
  if (!innerDimsPosArr.empty()) {
    innerDimsPos[0] = innerDimsPosArr[0];
    innerDimsPos[1] = innerDimsPosArr[1];
  }

  int64_t outerDimsPerm[2] = {0, 1};
  ArrayRef<int64_t> outerDimsPosArr = op.getOuterDimsPerm();
  if (!outerDimsPosArr.empty()) {
    outerDimsPerm[0] = outerDimsPosArr[0];
    outerDimsPerm[1] = outerDimsPosArr[1];
  }

  int flags = 0;

  if (innerDimsPos[0] == 0 && innerDimsPos[1] == 1) {
    // nothing to do
  } else if (innerDimsPos[0] == 1 && innerDimsPos[1] == 0) {
    flags |= IREE_UK_FLAG_PACK_TRANSPOSE_INNER;
  } else {
    return rewriter.notifyMatchFailure(op, "unsupported inner_dims_pos");
  }

  if (outerDimsPerm[0] == 0 && outerDimsPerm[1] == 1) {
    // nothing to do
  } else if (outerDimsPerm[0] == 1 && outerDimsPerm[1] == 0) {
    flags |= IREE_UK_FLAG_PACK_TRANSPOSE_OUTER;
  } else {
    return rewriter.notifyMatchFailure(op, "unsupported outer_dims_perm");
  }

  Location loc = op.getLoc();
  Value in_size0 = rewriter.create<tensor::DimOp>(loc, in, 0);
  Value in_size1 = rewriter.create<tensor::DimOp>(loc, in, 1);
  Value in_size2 = rewriter.create<tensor::DimOp>(loc, in, 2);
  Value in_size3 = rewriter.create<tensor::DimOp>(loc, in, 3);
  Value out_size0 = rewriter.create<tensor::DimOp>(loc, out, 0);
  Value out_size1 = rewriter.create<tensor::DimOp>(loc, out, 1);
  Value flagsVal = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(flags));
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fnName, in, out,
      ValueRange{in_size0, in_size1, in_size2, in_size3, out_size0, out_size1,
                 flagsVal},
      /*strided_outer_dims=*/rewriter.getIndexAttr(1));
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

namespace {

/// Pattern to lower (linalg.fill -> )? linalg.matmul operation sequence and
/// converts it into a iree_codegen.ukernel.generic operation
struct UKernelMatmulPattern : OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<IREE::Codegen::UKernelOpInterface> microKernelOp =
        matchDAGForUKernel(rewriter, matmulOp);
    if (failed(microKernelOp)) {
      return rewriter.notifyMatchFailure(
          matmulOp, "failed to find microkernel op to replace with");
    }
    rewriter.replaceOp(matmulOp, microKernelOp.value()->getResults());
    return success();
  }
};

/// Pattern to lower (linalg.fill -> )? linalg.mmt4d to ukernel.generic
struct UKernelMmt4DPattern : OpRewritePattern<linalg::Mmt4DOp> {
  using OpRewritePattern<linalg::Mmt4DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Mmt4DOp mmt4dOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<IREE::Codegen::UKernelOpInterface> microKernelOp =
        matchDAGForUKernel(rewriter, mmt4dOp);
    if (failed(microKernelOp)) {
      return rewriter.notifyMatchFailure(
          mmt4dOp, "failed to find microkernel op to replace with");
    }
    rewriter.replaceOp(mmt4dOp, microKernelOp.value()->getResults());
    return success();
  }
};

/// Pattern to lower tensor.pack to ukernel.generic
struct UKernelPackPattern : OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<IREE::Codegen::UKernelOpInterface> microKernelOp =
        matchDAGForUKernel(rewriter, packOp);
    if (failed(microKernelOp)) {
      return rewriter.notifyMatchFailure(
          packOp, "failed to find microkernel op to replace with");
    }
    rewriter.replaceOp(packOp, microKernelOp.value()->getResults());
    return success();
  }
};

/// Pattern to lower tensor.unpack to ukernel.generic
struct UKernelUnpackPattern : OpRewritePattern<tensor::UnPackOp> {
  using OpRewritePattern<tensor::UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::UnPackOp unpackOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<IREE::Codegen::UKernelOpInterface> microKernelOp =
        matchDAGForUKernel(rewriter, unpackOp);
    if (failed(microKernelOp)) {
      return rewriter.notifyMatchFailure(
          unpackOp, "failed to find microkernel op to replace with");
    }
    rewriter.replaceOp(unpackOp, microKernelOp.value()->getResults());
    return success();
  }
};

}  // namespace

void LLVMCPULowerToUKernelsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<UKernelMatmulPattern, UKernelMmt4DPattern, UKernelPackPattern,
                  UKernelUnpackPattern>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<>> createLLVMCPULowerToUKernelsPass() {
  return std::make_unique<LLVMCPULowerToUKernelsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
