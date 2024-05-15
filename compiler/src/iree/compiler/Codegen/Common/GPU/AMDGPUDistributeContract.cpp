// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/VectorOpUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#define DEBUG_TYPE "iree-codegen-amdgpu-distribute-contract"

namespace mlir::iree_compiler {
namespace {

using namespace mlir::iree_compiler::IREE::VectorExt;
using VectorValue = TypedValue<VectorType>;

/// Distributes `vector.contract` ops with nested layouts.
struct DistributeContract final : OpDistributionPattern<vector::ContractionOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<VectorType>(contractOp.getResultType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          contractOp, "unhandled contraction to scalar value");
    }

    auto resultValue = cast<VectorValue>(contractOp.getResult());
    NestedLayoutAttr resultLayout =
        dyn_cast<NestedLayoutAttr>(signature[resultValue]);
    if (!resultLayout) {
      return rewriter.notifyMatchFailure(
          contractOp, "missing nested layout for contraction result");
    }
    int64_t rank = resultLayout.getRank();

    NestedLayoutAttr lhsLayout =
        dyn_cast<NestedLayoutAttr>(signature[contractOp.getLhs()]);
    if (!lhsLayout) {
      return rewriter.notifyMatchFailure(
          contractOp, "missing nested layout for contraction lhs");
    }
    NestedLayoutAttr rhsLayout =
        dyn_cast<NestedLayoutAttr>(signature[contractOp.getRhs()]);
    if (!rhsLayout) {
      return rewriter.notifyMatchFailure(
          contractOp, "missing nested layout for contraction rhs");
    }

    // We assume there is an decision made before regarding which mfma intrinsic
    // to use and it is attached as an attribute to this contract op.
    auto mmaAttr =
        contractOp->getAttrOfType<IREE::GPU::MMAAttr>("iree.amdgpu.mma");
    if (!mmaAttr) {
      return rewriter.notifyMatchFailure(
          contractOp, "missing iree.amdgpu.mma intrinsic attribute");
    }

    // Infer the contract kind so that we know know to correlate M/N/K dims.
    VectorContractOpInfo opDetail(contractOp);
    if (opDetail.getOpKind() == VectorContractOpInfo::OpKind::UNKNOWN) {
      return rewriter.notifyMatchFailure(contractOp, "unknown contract kind");
    }

    SmallVector<int64_t> distShape = resultLayout.getDistributedShape();
    LLVM_DEBUG({
      llvm::dbgs() << "distributed shape: [";
      llvm::interleaveComma(distShape, llvm::dbgs());
      llvm::dbgs() << "]\n";
    });

    // Create a zero vector with the full distributed vector shape for
    // accumulating unrolled contraction results.
    auto tileType = VectorType::get(distShape, resultType.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(
        contractOp.getLoc(), tileType, rewriter.getZeroAttr(tileType));
    VectorValue finalTile = cast<VectorValue>(zero);
    LLVM_DEBUG(llvm::dbgs() << "init tile: " << finalTile << "\n");

    // Offsets into the LHS/RHS batches.
    SmallVector<int64_t> lhsBatchOffsets(rank, 0);
    SmallVector<int64_t> rhsBatchOffsets(rank, 0);

    // Offsets into the result batches.
    ArrayRef<int64_t> resultBatches = resultLayout.getBatchesPerSubgroup();
    SmallVector<int64_t> resultBatchTileSizes(rank, 1);
    LLVM_DEBUG({
      llvm::dbgs() << "result batches: [";
      llvm::interleaveComma(resultBatches, llvm::dbgs());
      llvm::dbgs() << "]\n";
    });

    Value acc = getDistributed(rewriter, cast<VectorValue>(contractOp.getAcc()),
                               resultLayout);
    Value lhs = getDistributed(rewriter, contractOp.getLhs(), lhsLayout);
    Value rhs = getDistributed(rewriter, contractOp.getRhs(), rhsLayout);

    SmallVector<AffineMap> indexingMaps = contractOp.getIndexingMapsArray();
    AffineMap lhsMap = compressUnusedDims(indexingMaps[0]);
    AffineMap rhsMap = compressUnusedDims(indexingMaps[1]);
    AffineMap resMap = compressUnusedDims(indexingMaps[2]);

    SmallVector<int64_t> resBatchOrder(resMap.getNumResults());
    std::iota(resBatchOrder.begin(), resBatchOrder.end(), 0);
    resBatchOrder = applyPermutationMap(resMap, ArrayRef(resBatchOrder));

    // Iterate over all result batches and unroll computation to direct MFMA
    // intrinsic ops.
    Location loc = contractOp.getLoc();
    auto resultTiles = StaticTileOffsetRange(
        resultBatches, resultBatchTileSizes, resBatchOrder);
    SmallVector<int64_t, 2> resultBatchOffsets;
    for (SmallVector<int64_t, 2> resultBatchOffsets : resultTiles) {
      LLVM_DEBUG({
        llvm::dbgs() << "current result batch offsets: [";
        llvm::interleaveComma(resultBatchOffsets, llvm::dbgs());
        llvm::dbgs() << "]\n";
      });

      // Get the slice of the accumulator in this batch.
      Value accSlice =
          rewriter.create<vector::ExtractOp>(loc, acc, resultBatchOffsets);

      // Get the k batch size for LHS and RHS vector.
      std::optional<int64_t> kBatch =
          getKBatchSize(opDetail, lhsLayout, rhsLayout);
      LLVM_DEBUG(llvm::dbgs() << "k batch size = " << kBatch << "\n");
      if (!kBatch) {
        return rewriter.notifyMatchFailure(contractOp,
                                           "A/B vector k batch mismatch");
      }

      // Perform contraction by doing separate outer product with amdgpu.mfma
      // operation and accumulate to the same vector.
      for (int k = 0; k < kBatch; ++k) {
        // Fills the batch offsets for LHS and RHS. For the K dimension it's the
        // induction variable; for the M/N dimension we need to extract from the
        // result batch offsets.
        fillOperandBatchOffsets(opDetail, k, resultBatchOffsets,
                                lhsBatchOffsets, rhsBatchOffsets, lhsMap,
                                rhsMap);
        LLVM_DEBUG({
          llvm::dbgs() << "current lhs batch offsets: [";
          llvm::interleaveComma(lhsBatchOffsets, llvm::dbgs());
          llvm::dbgs() << "]\n";
          llvm::dbgs() << "current rhs batch offsets: [";
          llvm::interleaveComma(rhsBatchOffsets, llvm::dbgs());
          llvm::dbgs() << "]\n";
        });

        Value lhsSlice =
            rewriter.create<vector::ExtractOp>(loc, lhs, lhsBatchOffsets);
        Value rhsSlice =
            rewriter.create<vector::ExtractOp>(loc, rhs, rhsBatchOffsets);
        accSlice =
            computeMMA(rewriter, loc, mmaAttr, lhsSlice, rhsSlice, accSlice);
      }
      finalTile = rewriter.create<vector::InsertOp>(loc, accSlice, finalTile,
                                                    resultBatchOffsets);
    }

    replaceOpWithDistributedValues(rewriter, contractOp, finalTile);
    return success();
  }

  // Gets the batch size for matmul K dimensions.
  std::optional<int64_t> getKBatchSize(const VectorContractOpInfo &opDetail,
                                       NestedLayoutAttr lhsLayout,
                                       NestedLayoutAttr rhsLayout) const {
    auto [lhsK, rhsK] = opDetail.getOperandKIndex();
    int64_t lhsKBatch = lhsLayout.getBatchesPerSubgroup()[lhsK];
    int64_t rhsKBatch = rhsLayout.getBatchesPerSubgroup()[rhsK];

    if (lhsKBatch != rhsKBatch)
      return std::nullopt;
    return lhsKBatch;
  }

  // Given a contract op's batch |resultOffsets|, fills its batch offsets for
  // both LHS and RHS.
  void fillOperandBatchOffsets(const VectorContractOpInfo &opDetail,
                               int64_t kOffset, ArrayRef<int64_t> resultOffsets,
                               SmallVector<int64_t> &lhsOffsets,
                               SmallVector<int64_t> &rhsOffsets,
                               AffineMap lhsMap, AffineMap rhsMap) const {
    auto [lhsK, rhsK] = opDetail.getOperandKIndex();
    // resultOffsets contains batch indices into the C/D vector. It is a 2-D
    // index for both M and N. We need to split out for M and N, and add index
    // for K.
    for (auto [lhsM, resultM] :
         llvm::zip_equal(opDetail.lhsMDims, opDetail.outMDims)) {
      lhsOffsets[lhsM] = resultOffsets[resultM];
    }

    if (opDetail.getBatchCount() == 1) {
      rhsOffsets[0] = resultOffsets[0];
      lhsOffsets[0] = resultOffsets[0];
    }

    for (auto [rhsN, resultN] :
         llvm::zip_equal(opDetail.rhsNDims, opDetail.outNDims)) {
      rhsOffsets[rhsN] = resultOffsets[resultN];
    }

    lhsOffsets[lhsK] = kOffset;
    rhsOffsets[rhsK] = kOffset;
  }

  // Generates amdgpu.mfma operation on the given inputs for the given MFMA
  // |intrinsic|.
  Value computeMMA(OpBuilder &builder, Location loc, IREE::GPU::MMAAttr mmaAttr,
                   Value a, Value b, Value c) const {
    // Get the storage vector types that each thread is in charge of.
    auto [aVectorType, bVectorType, cVectorType] = mmaAttr.getABCVectorTypes();
    Value aCast =
        builder.create<vector::ShapeCastOp>(a.getLoc(), aVectorType, a);
    Value bCast =
        builder.create<vector::ShapeCastOp>(b.getLoc(), bVectorType, b);
    Value cCast =
        builder.create<vector::ShapeCastOp>(c.getLoc(), cVectorType, c);
    FailureOr<Value> mmaOp = mmaAttr.buildMmaOperation(
        builder, loc, cVectorType, aCast, bCast, cCast);
    assert(succeeded(mmaOp) && "Failed to construct mma op");
    return builder.create<vector::ShapeCastOp>(c.getLoc(), c.getType(), *mmaOp);
  }
};

} // namespace

void populateGPUDistributeNestedLayoutContractAMDGPUPatterns(
    RewritePatternSet &patterns) {
  patterns.add<DistributeContract>(patterns.getContext());
}

} // namespace mlir::iree_compiler
