// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

//===----------------------------------------------------------------------===//
// Utils.
//===----------------------------------------------------------------------===//

/// Returns the size and offset scaled by some scale factor, and clamped to a
/// dimSize for the dimension. `(offset + size) * scale` will be clamped to the
/// `dimSize`.
static std::pair<OpFoldResult, OpFoldResult>
getScaledSizeAndOffset(OpBuilder &builder, Location loc, OpFoldResult size,
                       OpFoldResult offset, OpFoldResult dimSize,
                       int64_t offsetScale, int64_t sizeScale) {
  AffineExpr dim0, dim1, dim2;
  auto ctx = builder.getContext();
  bindDims(ctx, dim0, dim1, dim2);
  auto imageOffset = affine::makeComposedFoldedAffineApply(
      builder, loc, {dim0 * offsetScale}, offset);
  auto dimSizeValue = getValueOrCreateConstantIndexOp(builder, loc, dimSize);
  AffineMap sizeMap =
      AffineMap::get(3, 0, {dim0 - dim1, dim2 * sizeScale}, ctx);
  auto imageSize = affine::makeComposedFoldedAffineMin(
      builder, loc, sizeMap, {dimSizeValue, imageOffset, size});
  return std::make_pair(imageSize, imageOffset);
}

/// If the input has a fully static shape, return the static sizes. Otherwise,
/// attempt to reify the shape of the input from its defining op. Input dims
/// are store into `reifiedInputDims`.
static LogicalResult
getStaticOrReifiedInputDims(OpBuilder &builder, Location loc, Value input,
                            ReifiedRankedShapedTypeDims &reifiedInputDims) {
  if (auto reifyOp = input.getDefiningOp<ReifyRankedShapedTypeOpInterface>()) {
    return reifyOp.reifyResultShapes(builder, reifiedInputDims);
  }
  auto inputType = cast<ShapedType>(input.getType());
  if (!inputType.hasStaticShape()) {
    return failure();
  }
  reifiedInputDims.push_back(
      getAsIndexOpFoldResult(builder.getContext(), inputType.getShape()));
  return success();
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> ScatterOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getUpdateType().getRank(),
                                                 utils::IteratorType::parallel);
  if (!getUniqueIndices()) {
    iteratorTypes[0] = utils::IteratorType::reduction;
  }
  return iteratorTypes;
}

SmallVector<Range> ScatterOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Range> ranges;
  for (auto dim : llvm::seq<int64_t>(0, getUpdateType().getRank())) {
    Value ub = getDimValue(builder, loc, getUpdates(), dim);
    ranges.emplace_back(Range{zero, ub, one});
  }
  return ranges;
}

FailureOr<TilingResult>
ScatterOp::getTiledImplementation(OpBuilder &builder,
                                  ArrayRef<OpFoldResult> offsets,
                                  ArrayRef<OpFoldResult> sizes) {
  assert(offsets.size() >= 1 && sizes.size() >= 1);
  Location loc = getLoc();
  auto zeroAttr = builder.getI64IntegerAttr(0);
  auto oneAttr = builder.getI64IntegerAttr(1);

  // Slice of the updates.
  auto updateRank = getUpdateType().getRank();
  SmallVector<OpFoldResult> updateStrides(updateRank, oneAttr);
  Value tiledUpdate =
      getSlice(builder, loc, getUpdates(), offsets, sizes, updateStrides);
  assert(tiledUpdate && "failed to get slice of update");

  // Slice of indices.
  auto indicesRank = getIndicesType().getRank();
  SmallVector<OpFoldResult> indicesOffsets(indicesRank, zeroAttr);
  SmallVector<OpFoldResult> indicesSizes(indicesRank);
  indicesOffsets[0] = offsets[0];
  indicesSizes[0] = sizes[0];
  for (auto dim : llvm::seq<int64_t>(1, indicesRank)) {
    indicesSizes[dim] = getDim(builder, loc, getIndices(), dim);
  }
  SmallVector<OpFoldResult> indicesStrides(indicesRank, oneAttr);
  Value tiledIndices = getSlice(builder, loc, getIndices(), indicesOffsets,
                                indicesSizes, indicesStrides);
  assert(tiledIndices && "failed to get slice of indices");

  // Slice of the original.
  SmallVector<OpFoldResult> originalOffsets, originalSizes;
  if (failed(getResultTilePosition(builder, 0, offsets, sizes, originalOffsets,
                                   originalSizes))) {
    return {};
  }
  auto originalRank = getOriginalType().getRank();
  SmallVector<OpFoldResult> originalStrides(originalRank, oneAttr);
  Value tiledOriginal = getSlice(builder, loc, getOriginal(), originalOffsets,
                                 originalSizes, originalStrides);
  assert(tiledOriginal && "failed to get slice of original tensor");

  SmallVector<Type> resultTypes;
  if (getNumResults()) {
    resultTypes.push_back(tiledOriginal.getType());
  }
  Operation *tiledScatterOp =
      mlir::clone(builder, getOperation(), resultTypes,
                  ValueRange{tiledUpdate, tiledIndices, tiledOriginal});
  return TilingResult{{tiledScatterOp},
                      SmallVector<Value>(tiledScatterOp->getResults())};
}

LogicalResult ScatterOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  auto zeroAttr = builder.getI64IntegerAttr(0);
  // Slice of the original.
  auto originalRank = getOriginalType().getRank();
  resultOffsets.resize(originalRank, zeroAttr);
  resultSizes.resize(originalRank);

  auto updateRank = getUpdateType().getRank();
  Location loc = getLoc();
  for (auto dim : llvm::seq<int64_t>(0, originalRank - updateRank + 1)) {
    resultSizes[dim] = getDim(builder, loc, getOriginal(), dim);
  }
  for (auto dim :
       llvm::seq<int64_t>(originalRank - updateRank + 1, originalRank)) {
    resultOffsets[dim] = offsets[dim - (originalRank - updateRank)];
    resultSizes[dim] = sizes[dim - (originalRank - updateRank)];
  }
  return success();
}

LogicalResult ScatterOp::generateScalarImplementation(OpBuilder &b,
                                                      Location loc,
                                                      ValueRange ivs) {
  auto indexDepth = getIndexDepth();
  Value update = b.create<memref::LoadOp>(loc, getUpdates(), ivs);
  SmallVector<Value> starts;
  SmallVector<Value> loadIndices;
  loadIndices.push_back(ivs.front());
  loadIndices.push_back(Value());

  // Populate with empty values.
  auto originalTy = cast<ShapedType>(getOriginal().getType());
  starts.resize(originalTy.getRank(), Value());
  auto updateIvs = ivs.drop_front(1);

  int64_t offset = starts.size() - updateIvs.size();
  for (auto [idx, iv] : llvm::enumerate(updateIvs)) {
    starts[idx + offset] = iv;
  }

  ArrayRef<int64_t> dimMap = getDimensionMap();

  for (auto i : llvm::seq<unsigned>(0, indexDepth)) {
    loadIndices.back() = b.create<arith::ConstantIndexOp>(loc, i);
    Value idx = b.create<memref::LoadOp>(loc, getIndices(), loadIndices);
    Value ret = b.create<arith::IndexCastOp>(loc, b.getIndexType(), idx);

    auto dim = dimMap[i];

    if (starts[dim])
      ret = b.create<arith::AddIOp>(loc, ret, starts[dim]);
    starts[dim] = ret;
  }

  Value init = b.create<memref::LoadOp>(loc, getOriginal(), starts);

  IRMapping bvm;
  Block &block = getRegion().front();
  bvm.map(block.getArgument(0), update);
  bvm.map(block.getArgument(1), init);
  for (auto &blockOp : block.without_terminator()) {
    b.clone(blockOp, bvm);
  }
  // The last op is linalg_ext.yield op. Store the operand to
  // destination.
  b.create<memref::StoreOp>(
      loc, bvm.lookupOrDefault(block.getTerminator()->getOperand(0)),
      getOriginal(), starts);
  return success();
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> SortOp::getLoopIteratorTypes() {
  // All loops except the dimension to sort along are parallel.
  SmallVector<utils::IteratorType> iteratorTypes(getOperandRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getDimension()] = utils::IteratorType::reduction;
  return iteratorTypes;
}

SmallVector<Range> SortOp::getIterationDomain(OpBuilder &builder) {
  int64_t operandRank = getOperandRank();
  SmallVector<Range> loopBounds(operandRank);
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = getOperand(0);
  for (auto dim : llvm::seq<int64_t>(0, operandRank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = getDimValue(builder, loc, source, dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

FailureOr<TilingResult>
SortOp::getTiledImplementation(OpBuilder &builder,
                               ArrayRef<OpFoldResult> offsets,
                               ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getOperandRank();
  assert(offsets.size() == static_cast<size_t>(rank) &&
         sizes.size() == static_cast<size_t>(rank));
  auto oneAttr = builder.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> strides(rank, oneAttr);
  SmallVector<Value> tiledOperands(getOutputs().size());
  for (auto [idx, output] : llvm::enumerate(getOutputs())) {
    tiledOperands[idx] =
        getSlice(builder, getLoc(), output, offsets, sizes, strides);
    assert(tiledOperands[idx] && "failed to get slice of operand");
  }
  SmallVector<Type, 4> resultTypes;
  if (getNumResults()) {
    resultTypes = llvm::map_to_vector<4>(tiledOperands,
                                         [&](Value v) { return v.getType(); });
  }
  Operation *tiledSortOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);
  return TilingResult{{tiledSortOp},
                      SmallVector<Value>{tiledSortOp->getResults()}};
}

LogicalResult SortOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets = llvm::to_vector(offsets);
  resultSizes = llvm::to_vector(sizes);
  return success();
}

LogicalResult SortOp::generateScalarImplementation(OpBuilder &b, Location loc,
                                                   ValueRange ivs) {
  auto sortDim = getDimension();
  SmallVector<Value> indices, sortBlkArgs;
  indices.append(ivs.begin(), ivs.end());
  // Bubble sort innermost loop.
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value ub;
  if (getOperandType(0).isDynamicDim(sortDim)) {
    ub = b.create<memref::DimOp>(loc, getOperand(0), sortDim);
  } else {
    ub = b.create<arith::ConstantIndexOp>(
        loc, getOperandType(0).getDimSize(sortDim));
  }
  ub = b.create<arith::SubIOp>(loc, ub, one);
  auto scfFor = b.create<scf::ForOp>(
      loc, zero, ub, one, ValueRange{},
      [&](OpBuilder &b, Location loc, Value iv, ValueRange iters) {
        SmallVector<Value> indices(ivs);
        Value ivPlusOne = b.create<arith::AddIOp>(loc, iv, one);
        for (auto output : getDpsInits()) {
          indices[sortDim] = iv;
          sortBlkArgs.push_back(b.create<memref::LoadOp>(loc, output, indices));
          indices[sortDim] = ivPlusOne;
          sortBlkArgs.push_back(b.create<memref::LoadOp>(loc, output, indices));
        }
      });

  auto &srcBlock = getRegion().front();
  Region &region = scfFor.getRegion();
  IRMapping bvm;
  {
    OpBuilder::InsertionGuard guard(b);
    auto &block = region.front();
    b.setInsertionPointToEnd(&block);
    for (auto it : llvm::zip_equal(srcBlock.getArguments(), sortBlkArgs)) {
      bvm.map(std::get<0>(it), std::get<1>(it));
    }
    for (auto &blockOp : srcBlock.without_terminator()) {
      b.clone(blockOp, bvm);
    }
  }
  Value cond = bvm.lookupOrDefault(srcBlock.getTerminator()->getOperand(0));

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToEnd(&region.front());
  b.create<scf::IfOp>(
      loc, cond,
      [&](OpBuilder &b, Location loc) {
        // Do not swap the pairs if true.
        b.create<scf::YieldOp>(loc);
      },
      [&](OpBuilder &b, Location loc) {
        // Swap the pairs if false.
        SmallVector<Value> indices(ivs.begin(), ivs.end());
        Value ivPlusOne =
            b.create<arith::AddIOp>(loc, scfFor.getInductionVar(), one);
        for (int i = 0, e = getNumDpsInits(); i < e; ++i) {
          Value v1 = sortBlkArgs[i * 2];
          Value v2 = sortBlkArgs[i * 2 + 1];
          indices[sortDim] = scfFor.getInductionVar();
          b.create<memref::StoreOp>(loc, v2, getDpsInits()[i], indices);
          indices[sortDim] = ivPlusOne;
          b.create<memref::StoreOp>(loc, v1, getDpsInits()[i], indices);
        }
        b.create<scf::YieldOp>(loc);
      });
  b.create<scf::YieldOp>(loc);
  return success();
}

//===----------------------------------------------------------------------===//
// FftOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> FftOp::getLoopIteratorTypes() {
  // There are `rank-1` outer loops. The fft itselfs has one loop for each
  // stage, which handles the merge step -- taking two half size tensors and
  // merge them into one tensor.
  SmallVector<utils::IteratorType> iteratorTypes(getOperandRank(),
                                                 utils::IteratorType::parallel);
  return iteratorTypes;
}

SmallVector<Range> FftOp::getIterationDomain(OpBuilder &builder) {
  SmallVector<Range> res;
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  for (auto [idx, val] : llvm::enumerate(getOperandShape().drop_back())) {
    Value size;
    if (ShapedType::isDynamic(val)) {
      size = getDimValue(builder, loc, getReal(), idx);
    } else {
      size = builder.create<arith::ConstantIndexOp>(loc, val);
    }
    res.emplace_back(Range{/*offset=*/zero, size, /*stride=*/one});
  }

  Value size = getDimValue(builder, loc, getReal(), getOperandRank() - 1);
  Value stride = builder.create<arith::ShLIOp>(loc, one, getStage());
  res.emplace_back(Range{/*offset=*/zero, size, /*stride=*/stride});
  return res;
}

void FftOp::generateScalarImplWithoutCoeffBuf(OpBuilder &b, Location loc,
                                              ArrayRef<Value> operands,
                                              Value wholeSize) {
  auto rank = getOperandRank();
  SmallVector<AffineMap> maps(operands.size(), b.getMultiDimIdentityMap(rank));

  auto f32Type = b.getF32Type();
  auto indexToF32 = [](OpBuilder &builder, Location loc, Value v) -> Value {
    v = builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), v);
    return builder.create<arith::SIToFPOp>(loc, builder.getF32Type(), v);
  };

  // We will need exp(-2 * PI * j / m * I), compute "-2 * PI / m" for imag part
  // first.
  Value coeff = b.create<arith::ConstantFloatOp>(
      loc, llvm::APFloat(static_cast<float>(-2 * acos(-1))), f32Type);
  coeff = b.create<arith::DivFOp>(loc, coeff, indexToF32(b, loc, wholeSize));

  b.create<linalg::GenericOp>(
      loc, TypeRange{}, ValueRange{}, operands, maps, getLoopIteratorTypes(),
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value lhsReal = args[0];
        Value lhsImag = args[1];
        Value rhsReal = args[2];
        Value rhsImag = args[3];

        // Compute "-2 * PI / m * j"
        Value w = b.create<arith::MulFOp>(
            loc, coeff,
            indexToF32(b, loc, b.create<linalg::IndexOp>(loc, rank - 1)));
        Value wReal = b.create<math::CosOp>(loc, w);
        Value wImag = b.create<math::SinOp>(loc, w);

        // t = w * a[k + j + mh];
        // ->  (x + yi)(u + vi) = (xu - yv) + (xv + yu)i
        Value xu = b.create<arith::MulFOp>(loc, wReal, rhsReal);
        Value yv = b.create<arith::MulFOp>(loc, wImag, rhsImag);
        Value xv = b.create<arith::MulFOp>(loc, wReal, rhsImag);
        Value yu = b.create<arith::MulFOp>(loc, wImag, rhsReal);
        Value tReal = b.create<arith::SubFOp>(loc, xu, yv);
        Value tImag = b.create<arith::AddFOp>(loc, xv, yu);

        // cplx u = a[k + j];
        // a[k + j] = u + t;
        // a[k + j + mh] = u - t;
        Value r1 = b.create<arith::AddFOp>(loc, lhsReal, tReal);
        Value r2 = b.create<arith::AddFOp>(loc, lhsImag, tImag);
        Value r3 = b.create<arith::SubFOp>(loc, lhsReal, tReal);
        Value r4 = b.create<arith::SubFOp>(loc, lhsImag, tImag);
        b.create<linalg::YieldOp>(loc, ValueRange{r1, r2, r3, r4});
      });
}

void FftOp::generateScalarImplWithCoeffBuf(OpBuilder &b, Location loc,
                                           ArrayRef<Value> operands) {
  auto rank = getOperandRank();
  SmallVector<AffineMap> maps;
  // The size of coefficent buffer is epxected to match `2^(stage-1)`, which
  // equals to the last dim of operands.
  maps.append(
      2, AffineMap::get(rank, 0, b.getAffineDimExpr(rank - 1), b.getContext()));
  maps.append(operands.size(), b.getMultiDimIdentityMap(rank));

  b.create<linalg::GenericOp>(
      loc, TypeRange{}, ValueRange{getRealCoeff(), getImagCoeff()}, operands,
      maps, getLoopIteratorTypes(),
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value wReal = args[0];
        Value wImag = args[1];
        Value lhsReal = args[2];
        Value lhsImag = args[3];
        Value rhsReal = args[4];
        Value rhsImag = args[5];

        // t = w * a[k + j + mh];
        // ->  (x + yi)(u + vi) = (xu - yv) + (xv + yu)i
        Value xu = b.create<arith::MulFOp>(loc, wReal, rhsReal);
        Value yv = b.create<arith::MulFOp>(loc, wImag, rhsImag);
        Value xv = b.create<arith::MulFOp>(loc, wReal, rhsImag);
        Value yu = b.create<arith::MulFOp>(loc, wImag, rhsReal);
        Value tReal = b.create<arith::SubFOp>(loc, xu, yv);
        Value tImag = b.create<arith::AddFOp>(loc, xv, yu);

        // cplx u = a[k + j];
        // a[k + j] = u + t;
        // a[k + j + mh] = u - t;
        Value r1 = b.create<arith::AddFOp>(loc, lhsReal, tReal);
        Value r2 = b.create<arith::AddFOp>(loc, lhsImag, tImag);
        Value r3 = b.create<arith::SubFOp>(loc, lhsReal, tReal);
        Value r4 = b.create<arith::SubFOp>(loc, lhsImag, tImag);
        b.create<linalg::YieldOp>(loc, ValueRange{r1, r2, r3, r4});
      });
}

// Generates FFT stage scalar implementation. This follows Cooley–Tukey FFT
// algorithm. The pseudo reference code is:
//   let s <- stage of linalg_ext.fft
//   int m = 1 << s;
//   int mh = m >> 1;
//   for (int k = 0; k < n; k += m) {
//     for (int j = 0; j < mh; ++j) {
//       cplx w = exp(-2 * PI * j / m * I);
//       cplx t = w * a[k + j + mh];
//       cplx u = a[k + j];
//       a[k + j] = u + t;
//       a[k + j + mh] = u - t;
//     }
//   }
LogicalResult FftOp::generateScalarImplementation(OpBuilder &b, Location loc,
                                                  ValueRange ivs) {
  Value real = getReal();
  Value imag = getImag();
  Value stage = getStage();
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value wholeSize = b.create<arith::ShLIOp>(loc, one, stage);
  Value halfSize = b.create<arith::ShRSIOp>(loc, wholeSize, one);

  auto rank = getOperandRank();
  SmallVector<Value> operands;
  SmallVector<OpFoldResult> lhsIvs(ivs.begin(), ivs.end());
  SmallVector<OpFoldResult> ones(rank, b.getIndexAttr(1));
  SmallVector<OpFoldResult> sizes(rank, b.getIndexAttr(1));
  sizes.back() = halfSize;
  operands.push_back(
      b.create<memref::SubViewOp>(loc, real, lhsIvs, sizes, ones));
  operands.push_back(
      b.create<memref::SubViewOp>(loc, imag, lhsIvs, sizes, ones));

  SmallVector<OpFoldResult> rhsIvs(ivs.begin(), ivs.end());
  rhsIvs.back() =
      b.create<arith::AddIOp>(loc, ivs.back(), halfSize).getResult();
  operands.push_back(
      b.create<memref::SubViewOp>(loc, real, rhsIvs, sizes, ones));
  operands.push_back(
      b.create<memref::SubViewOp>(loc, imag, rhsIvs, sizes, ones));

  if (hasCoeff()) {
    generateScalarImplWithCoeffBuf(b, loc, operands);
  } else {
    generateScalarImplWithoutCoeffBuf(b, loc, operands, wholeSize);
  }

  return success();
}

FailureOr<TilingResult>
FftOp::getTiledImplementation(OpBuilder &builder,
                              ArrayRef<OpFoldResult> offsets,
                              ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getOperandRank();
  SmallVector<OpFoldResult> strides(rank, builder.getI64IntegerAttr(1));
  SmallVector<Value> tiledOperands(3);
  tiledOperands[0] = getStage();
  tiledOperands[1] = getRealCoeff();
  tiledOperands[2] = getImagCoeff();
  SmallVector<Type, 4> resultTypes;

  for (auto out : getOutputs()) {
    tiledOperands.push_back(
        getSlice(builder, getLoc(), out, offsets, sizes, strides));
    if (hasPureTensorSemantics()) {
      resultTypes.push_back(tiledOperands.back().getType());
    }
  }
  Operation *tiledFftOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);
  return TilingResult{{tiledFftOp},
                      SmallVector<Value>(tiledFftOp->getResults())};
}

LogicalResult FftOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  return success();
}

//===----------------------------------------------------------------------===//
// ScanOp
//===----------------------------------------------------------------------===//

SmallVector<Range> ScanOp::getIterationDomain(OpBuilder &builder) {
  int64_t operandRank = getOperandRank();
  SmallVector<Range> loopBounds(operandRank);
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = getInput();
  for (auto dim : llvm::seq<int64_t>(0, operandRank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = getDimValue(builder, loc, source, dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

SmallVector<utils::IteratorType> ScanOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getOperandRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getDimension()] = utils::IteratorType::reduction;
  return iteratorTypes;
}

// Generates naive scalar implementation of scan for a given operator f.
// For inclusive,
//     output[0] = input[0]
//     output[i] = f(output[i-1], input[i])
//
// For exclusive,
//     output[0] = 0
//     output[i] = f(output[i-1], input[i-1])

LogicalResult ScanOp::generateScalarImplementation(OpBuilder &b, Location loc,
                                                   ValueRange ivs) {
  SmallVector<Value> indices, scanBlkArgs;
  indices.append(ivs.begin(), ivs.end());
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  auto scanDim = getDimension();
  auto cond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                      indices[scanDim], zero);
  bool isInclusive = getInclusive();
  SmallVector<Value> accIndices;
  for (int i = 0; i < indices.size(); i++) {
    if (i != scanDim) {
      accIndices.push_back(indices[i]);
    }
  }

  auto scfIf = b.create<scf::IfOp>(
      loc, cond,
      [&](OpBuilder &b, Location loc) {
        if (isInclusive) {
          auto value = b.create<memref::LoadOp>(loc, getInput(), indices);
          b.create<memref::StoreOp>(loc, value, getOutput(), indices);
        } else {
          auto value =
              b.create<memref::LoadOp>(loc, getAccumulator(), accIndices);
          b.create<memref::StoreOp>(loc, value, getOutput(), indices);
        }
        b.create<scf::YieldOp>(loc);
      },
      [&](OpBuilder &b, Location loc) {
        SmallVector<Value> indices(ivs.begin(), ivs.end());
        Value iv = indices[scanDim];
        Value ivMinusOne = b.create<arith::SubIOp>(loc, iv, one);
        indices[scanDim] = ivMinusOne;
        scanBlkArgs.push_back(
            b.create<memref::LoadOp>(loc, getOutput(), indices));
        Value i0;
        if (!isInclusive)
          i0 = b.create<memref::LoadOp>(loc, getInput(), indices);
        indices[scanDim] = iv;
        if (isInclusive)
          i0 = b.create<memref::LoadOp>(loc, getInput(), indices);
        scanBlkArgs.push_back(i0);
      });

  auto &srcBlock = getRegion().front();
  Region &region = scfIf.getElseRegion();
  IRMapping bvm;
  {
    OpBuilder::InsertionGuard guard(b);
    auto &block = region.front();
    b.setInsertionPointToEnd(&block);
    for (auto it : llvm::zip_equal(srcBlock.getArguments(), scanBlkArgs)) {
      bvm.map(std::get<0>(it), std::get<1>(it));
    }
    for (auto &blockOp : srcBlock.without_terminator()) {
      b.clone(blockOp, bvm);
    }
    b.create<memref::StoreOp>(
        loc, bvm.lookupOrDefault(srcBlock.getTerminator()->getOperand(0)),
        getOutput(), indices);
    b.create<memref::StoreOp>(
        loc, bvm.lookupOrDefault(srcBlock.getTerminator()->getOperand(0)),
        getAccumulator(), accIndices);
    b.create<scf::YieldOp>(loc);
  }
  return success();
}

FailureOr<TilingResult>
ScanOp::getTiledImplementation(OpBuilder &builder,
                               ArrayRef<OpFoldResult> offsets,
                               ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getOperandRank();
  assert(offsets.size() == static_cast<size_t>(rank) &&
         sizes.size() == static_cast<size_t>(rank));
  auto oneAttr = builder.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> strides(rank, oneAttr);
  SmallVector<Value> tiledOperands;
  tiledOperands.emplace_back(
      getSlice(builder, getLoc(), getInput(), offsets, sizes, strides));
  tiledOperands.emplace_back(
      getSlice(builder, getLoc(), getOutputs()[0], offsets, sizes, strides));
  if (rank > 1) {
    SmallVector<OpFoldResult> accumOffsets, accumSizes;
    if (failed(getResultTilePosition(builder, 1, offsets, sizes, accumOffsets,
                                     accumSizes))) {
      return {};
    }
    SmallVector<OpFoldResult> accumStrides(rank - 1, oneAttr);
    tiledOperands.emplace_back(getSlice(builder, getLoc(), getOutputs()[1],
                                        accumOffsets, accumSizes,
                                        accumStrides));
  } else {
    tiledOperands.emplace_back(getOutputs()[1]);
  }

  SmallVector<Type, 4> resultTypes;
  if (hasPureTensorSemantics()) {
    resultTypes.push_back(tiledOperands[1].getType());
    resultTypes.push_back(tiledOperands[2].getType());
  }

  Operation *tiledScanOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);
  return TilingResult{{tiledScanOp},
                      SmallVector<Value>(tiledScanOp->getResults())};
}

LogicalResult ScanOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber == 0) {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }
  if (resultNumber == 1) {
    int64_t rank = getOperandRank();
    if (rank > 1) {
      for (auto i : llvm::seq<int64_t>(0, rank)) {
        if (i == getDimension())
          continue;
        resultOffsets.push_back(offsets[i]);
        resultSizes.push_back(sizes[i]);
      }
    }
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> ReverseOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getOperandRank(),
                                                 utils::IteratorType::parallel);
  return iteratorTypes;
}

SmallVector<Range> ReverseOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Range> ranges;
  for (auto dim : llvm::seq<int64_t>(0, getOperandRank())) {
    Value ub = getDimValue(builder, loc, getInput(), dim);
    ranges.emplace_back(Range{zero, ub, one});
  }
  return ranges;
}

LogicalResult ReverseOp::generateScalarImplementation(OpBuilder &b,
                                                      Location loc,
                                                      ValueRange ivs) {
  SmallVector<Value> mirrorIndices(ivs.begin(), ivs.end());
  for (auto dim : getDimensionsArray()) {
    auto size = getDimValue(b, loc, getInput(), dim);
    size = b.create<arith::SubIOp>(loc, size,
                                   b.create<arith::ConstantIndexOp>(loc, 1));
    mirrorIndices[dim] = b.create<arith::SubIOp>(loc, size, mirrorIndices[dim]);
  }
  Value val = b.create<memref::LoadOp>(loc, getInput(), ivs);
  b.create<memref::StoreOp>(loc, val, getOutput(), mirrorIndices);
  return success();
}

FailureOr<TilingResult>
ReverseOp::getTiledImplementation(OpBuilder &builder,
                                  ArrayRef<OpFoldResult> offsets,
                                  ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getOperandRank();
  SmallVector<OpFoldResult> strides(rank, builder.getI64IntegerAttr(1));
  Location loc = getLoc();
  SmallVector<OpFoldResult> mirrorOffsets, mirrorSizes;
  if (failed(getResultTilePosition(builder, 0, offsets, sizes, mirrorOffsets,
                                   mirrorSizes))) {
    return {};
  }

  SmallVector<Value> tiledOperands;
  tiledOperands.emplace_back(
      getSlice(builder, loc, getInput(), offsets, sizes, strides));

  SmallVector<Type, 4> resultTypes;
  if (hasPureTensorSemantics()) {
    tiledOperands.emplace_back(
        getSlice(builder, loc, getOutput(), mirrorOffsets, sizes, strides));
    resultTypes.push_back(tiledOperands[1].getType());
  } else {
    tiledOperands.emplace_back(
        getSlice(builder, loc, getOutput(), mirrorOffsets, sizes, strides));
  }

  Operation *tiledRevOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{{tiledRevOp},
                      SmallVector<Value>(tiledRevOp->getResults())};
}

LogicalResult ReverseOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  AffineExpr sym0, sym1, sym2;
  bindSymbols(builder.getContext(), sym0, sym1, sym2);
  AffineMap map =
      AffineMap::get(/*dimCount=*/0, /*symbolCount=*/3, {sym0 - sym1 - sym2});
  resultOffsets.assign(offsets.begin(), offsets.end());
  Location loc = getLoc();
  for (auto dim : getDimensionsArray()) {
    Value size = getDimValue(builder, loc, getInput(), dim);
    Value offset =
        getValueOrCreateConstantIndexOp(builder, loc, resultOffsets[dim]);
    Value tileSize = getValueOrCreateConstantIndexOp(builder, loc, sizes[dim]);
    resultOffsets[dim] = builder
                             .create<affine::AffineApplyOp>(
                                 loc, map, ValueRange{size, offset, tileSize})
                             .getResult();
  }
  resultSizes.assign(sizes.begin(), sizes.end());
  return success();
}

//===----------------------------------------------------------------------===//
// TopkOp
//===----------------------------------------------------------------------===//

SmallVector<Range> TopkOp::getIterationDomain(OpBuilder &builder) {
  int64_t operandRank = getInputRank();
  SmallVector<Range> loopBounds(operandRank);
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = getValues();
  for (auto [idx, val] : llvm::enumerate(getInputType().getShape())) {
    loopBounds[idx].offset = zero;
    loopBounds[idx].size = getDimValue(builder, loc, source, idx);
    loopBounds[idx].stride = one;
  }
  return loopBounds;
}

SmallVector<utils::IteratorType> TopkOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getInputRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getDimension()] = utils::IteratorType::reduction;
  return iteratorTypes;
}

LogicalResult TopkOp::generateScalarImplementation(OpBuilder &b, Location loc,
                                                   ValueRange ivs) {
  uint64_t kDim = getDimension();
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value initialValue = b.create<memref::LoadOp>(loc, getValues(), ivs);

  // If the indices tensor is not provided, the value index is derived from the
  // loop induction variables.
  Value initialIndex;
  if (getIndices()) {
    initialIndex = b.create<memref::LoadOp>(loc, *getIndices(), ivs);
  } else {
    Value rawInitialIndex = ivs[kDim];
    initialIndex =
        b.create<arith::IndexCastOp>(loc, b.getI32Type(), rawInitialIndex);
  }

  // Compute K (ub) from the selected dim of the output
  Value ub = b.create<memref::DimOp>(loc, outputValues(), getDimension());

  // Inner K loop functions:
  //   Load current K value and index
  //   Compare N/K using inserted block compare
  //   Check if N == K using strict weak ordering, select which index came first
  //   Select new K value from N/K comparison
  //   Select new K index from N/K comparison or which index came first
  //   Store new k value and index
  //   Yield loop carry values after K selection
  Value kValue, kIndex;
  auto scfFor = b.create<scf::ForOp>(
      loc, zero, ub, one, ValueRange{initialValue, initialIndex},
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopCarryValues) {
        SmallVector<Value> indices(ivs);
        indices[kDim] = iv;
        kValue = b.create<memref::LoadOp>(loc, outputValues(), indices);
        kIndex = b.create<memref::LoadOp>(loc, outputIndices(), indices);
      });

  SmallVector<Value> indices(ivs);
  indices[kDim] = scfFor.getInductionVar();
  auto loopCarryValues = scfFor.getRegionIterArgs();

  // Retrieve region as black box comparision function f(x,y). Plug into op.
  auto &srcBlock = getRegion().front();
  IRMapping bvmF; // f(x,y)
  IRMapping bvmR; // f(y,x)
  {
    // Save previous insertion point. Continue within loop body.
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(&scfFor.getRegion().front());
    SmallVector<Value> forwardValues{loopCarryValues[0], kValue};
    SmallVector<Value> reverseValues{kValue, loopCarryValues[0]};
    for (auto it : llvm::zip_equal(srcBlock.getArguments(), forwardValues)) {
      bvmF.map(std::get<0>(it), std::get<1>(it));
    }
    for (auto it : llvm::zip_equal(srcBlock.getArguments(), reverseValues)) {
      bvmR.map(std::get<0>(it), std::get<1>(it));
    }
    for (auto &blockOp : srcBlock.without_terminator()) {
      b.clone(blockOp, bvmF);
      b.clone(blockOp, bvmR);
    }
    Value forwardCmpRes = bvmF.lookup(srcBlock.getTerminator()->getOperand(0));
    Value reverseCmpRes = bvmR.lookup(srcBlock.getTerminator()->getOperand(0));

    // Check value equality using strictly weak ordering from the region:
    //   f(x,y) --> forwardCmpRes
    //   f(y,x) --> reverseCmpRes
    //   if forwardCmpRes == reverseCmpRes then select which came first
    Value cmpValuesEqual = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, forwardCmpRes, reverseCmpRes);
    Value cmpFirstIndex = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, loopCarryValues[1], kIndex);
    Value combinedCmpEqRes =
        b.create<arith::AndIOp>(loc, cmpValuesEqual, cmpFirstIndex);
    // True if N > K or N came before K
    Value indexCmpRes =
        b.create<arith::OrIOp>(loc, forwardCmpRes, combinedCmpEqRes);
    // Select results for K based on comparisons
    Value resultKValue = b.create<arith::SelectOp>(loc, forwardCmpRes,
                                                   loopCarryValues[0], kValue);
    Value resultKIndex =
        b.create<arith::SelectOp>(loc, indexCmpRes, loopCarryValues[1], kIndex);
    b.create<memref::StoreOp>(loc, resultKValue, outputValues(), indices);
    b.create<memref::StoreOp>(loc, resultKIndex, outputIndices(), indices);
    // Select loop carry, opposite of K results
    Value resultCarryValue = b.create<arith::SelectOp>(
        loc, forwardCmpRes, kValue, loopCarryValues[0]);
    Value resultCarryIndex =
        b.create<arith::SelectOp>(loc, indexCmpRes, kIndex, loopCarryValues[1]);
    b.create<scf::YieldOp>(loc, ValueRange{resultCarryValue, resultCarryIndex});
  }
  return success();
}

FailureOr<TilingResult>
TopkOp::getTiledImplementation(OpBuilder &builder,
                               ArrayRef<OpFoldResult> offsets,
                               ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getInputRank();
  assert(offsets.size() == static_cast<size_t>(rank) &&
         sizes.size() == static_cast<size_t>(rank));
  SmallVector<OpFoldResult> strides(rank, builder.getI64IntegerAttr(1));
  Location loc = getLoc();

  SmallVector<OpFoldResult> outputOffsets, outputSizes;
  if (failed(getResultTilePosition(builder, 0, offsets, sizes, outputOffsets,
                                   outputSizes))) {
    return {};
  }

  SmallVector<Value> tiledOperands;
  tiledOperands.emplace_back(
      getSlice(builder, loc, getValues(), offsets, sizes, strides));
  if (getIndices()) {
    tiledOperands.emplace_back(
        getSlice(builder, loc, *getIndices(), offsets, sizes, strides));
  }

  // Replace the tile size for the K dimension to use the output size instead of
  // the input size.
  Value kSize = getDimValue(builder, getLoc(), outputValues(), getDimension());
  outputSizes[getDimension()] = getAsOpFoldResult(kSize);

  tiledOperands.emplace_back(
      getSlice(builder, loc, getOutputs()[0], offsets, outputSizes, strides));
  tiledOperands.emplace_back(
      getSlice(builder, loc, getOutputs()[1], offsets, outputSizes, strides));
  SmallVector<Type, 2> resultTypes;
  if (hasPureTensorSemantics()) {
    resultTypes.push_back(tiledOperands[tiledOperands.size() - 2].getType());
    resultTypes.push_back(tiledOperands[tiledOperands.size() - 1].getType());
  }

  Operation *tiledTopkOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);
  return TilingResult{{tiledTopkOp},
                      SmallVector<Value>(tiledTopkOp->getResults())};
}

LogicalResult TopkOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  Value kSize = getDimValue(builder, getLoc(), getDpsInits()[resultNumber],
                            getDimension());
  resultSizes[getDimension()] = getAsOpFoldResult(kSize);
  return success();
}

//===----------------------------------------------------------------------===//
// PackOp and UnPackOp utils
//===----------------------------------------------------------------------===//

/// Utility function to build the iteration domain for `packOp` or `unPackOp`.
template <typename OpTy>
static SmallVector<Range> getIterationDomain(OpTy op, OpBuilder &builder) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  OpBuilder::InsertionGuard g(builder);
  Location loc = op.getLoc();
  int64_t rank = (std::is_same<OpTy, PackOp>::value) ? op.getInputRank()
                                                     : op.getOutputRank();
  SmallVector<Range> loopBounds(rank);
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  ReifiedRankedShapedTypeDims resultShape;
  (void)op.reifyResultShapes(builder, resultShape);
  for (auto dim : llvm::seq<int64_t>(0, rank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].stride = one;
    loopBounds[dim].size = resultShape[0][dim];
  }
  return loopBounds;
}

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

SmallVector<Range> PackOp::getIterationDomain(OpBuilder &builder) {
  return LinalgExt::getIterationDomain(*this, builder);
}

/// Generate the body of the innermost loop of the scalar implementation
/// of `pack` operation.
static void generatePackOpScalarImplementationBody(PackOp packOp,
                                                   OpBuilder &builder,
                                                   Location loc,
                                                   ValueRange ivs) {
  // Note: `ivs` are already in the correct order, possibly interchanged based
  // on `dims_pos`. However, connecting the loops with the access patterns is
  // difficult - What is the relation between the position of the tile loop and
  // the point loop? However, if we interchange `ivs` once more to go to the
  // canonical blocking format: ABCabc, this connection becomes trivial: Each
  // point loop is pointLoopsOffset + inputRank away from the tiled loop.
  ArrayRef<int64_t> dimsToInnerBlock = packOp.getInnerDimsPos();
  ArrayRef<int64_t> dimsToOuterBlock = packOp.getOuterDimsPerm();

  SmallVector<Value> interchangedIvs = ivs;
  SmallVector<int64_t> interchangeVector =
      computeInterchangeFromDimPos(dimsToInnerBlock, packOp.getInputRank());
  interchangedIvs = interchange<Value>(interchangedIvs, interchangeVector,
                                       /*offset=*/packOp.getInputRank());
  if (!dimsToOuterBlock.empty()) {
    interchangeVector =
        computeInterchangeFromDimPos(dimsToOuterBlock, packOp.getInputRank());
    interchangedIvs =
        interchange<Value>(interchangedIvs, interchangeVector, /*offset=*/0);
  }

  SmallVector<OpFoldResult> tiles = packOp.getMixedTiles();
  DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
      packOp.getDimAndTileMapping();
  SmallVector<OpFoldResult> sourceIndices;
  size_t pointLoopsOffset = 0;
  int64_t inputRank = packOp.getInputRank();
  for (auto dim : llvm::seq<int64_t>(0, inputRank)) {
    if (dimAndTileMapping.count(dim)) {
      AffineExpr i, j, tile;
      bindDims(builder.getContext(), i, j);
      bindSymbols(builder.getContext(), tile);
      OpFoldResult sourceIndex = affine::makeComposedFoldedAffineApply(
          builder, loc, i * tile + j,
          ArrayRef<OpFoldResult>{
              interchangedIvs[dim],
              interchangedIvs[pointLoopsOffset + packOp.getInputRank()],
              dimAndTileMapping[dim]});
      sourceIndices.push_back(sourceIndex);
      ++pointLoopsOffset;
    } else {
      sourceIndices.push_back(interchangedIvs[dim]);
    }
  }

  auto createLoad = [&]() -> Value {
    return builder.create<memref::LoadOp>(
        loc, packOp.getInput(),
        getValueOrCreateConstantIndexOp(builder, loc, sourceIndices));
  };
  Value scalar;
  if (auto paddingValue = packOp.getPaddingValue()) {
    ArithBuilder arithBuilder(builder, loc);
    Value isInBounds;
    for (auto dim : llvm::seq<int64_t>(0, inputRank)) {
      Value idx =
          getValueOrCreateConstantIndexOp(builder, loc, sourceIndices[dim]);
      Value cond = arithBuilder.slt(
          idx, getDimValue(builder, loc, packOp.getInput(), dim));
      isInBounds = dim == 0 ? cond : arithBuilder._and(isInBounds, cond);
    }
    scalar = builder
                 .create<scf::IfOp>(
                     loc, isInBounds, /*thenBuilder=*/
                     [&](OpBuilder &b, Location l) {
                       b.create<scf::YieldOp>(l, createLoad());
                     },
                     /*elseBuilder=*/
                     [&](OpBuilder &b, Location l) {
                       b.create<scf::YieldOp>(l, paddingValue);
                     })
                 .getResult(0);
  } else {
    scalar = createLoad();
  }

  builder.create<memref::StoreOp>(loc, scalar, packOp.getOutput(), ivs);
}

LogicalResult PackOp::generateScalarImplementation(OpBuilder &builder,
                                                   Location loc,
                                                   ValueRange ivs) {
  OpBuilder::InsertionGuard g(builder);
  // The `ivs` already represent the position into the output tensor for the
  // non data-tile dimensions.
  SmallVector<Value> ivVec = llvm::to_vector(ivs);
  ReifiedRankedShapedTypeDims outputShape;
  if (failed(reifyResultShapes(builder, outputShape))) {
    return getOperation()->emitOpError("failed to reify result shape");
  }
  if (outputShape.size() != 1 || outputShape[0].size() != getOutputRank()) {
    return getOperation()->emitOpError(
               "expected shape of one result value of rank")
           << getOutputRank();
  }

  // Generate the loops that iterate over the data tile.
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);

  // All loops except the innermost are simple loops that just iterate
  // over the tile dimensions.
  for (auto dataTileDim :
       llvm::seq<unsigned>(getInputRank(), getOutputRank() - 1)) {
    Value ub = getValueOrCreateConstantIndexOp(builder, loc,
                                               outputShape[0][dataTileDim]);
    scf::ForOp loop = builder.create<scf::ForOp>(loc, zero, ub, one);
    builder.setInsertionPointToStart(loop.getBody());
    ivVec.push_back(loop.getInductionVar());
  }
  // The body of the innermost loops does the actual data movement.
  builder.create<scf::ForOp>(
      loc, zero,
      getValueOrCreateConstantIndexOp(builder, loc, outputShape[0].back()), one,
      ValueRange{},
      [&](OpBuilder &bodyBuilder, Location bodyLoc, Value iv,
          ValueRange regionIterArgs) {
        ivVec.push_back(iv);
        generatePackOpScalarImplementationBody(*this, bodyBuilder, bodyLoc,
                                               ivVec);
        bodyBuilder.create<scf::YieldOp>(bodyLoc);
      });
  return success();
}

//===----------------------------------------------------------------------===//
// UnPackOp
//===----------------------------------------------------------------------===//

LogicalResult UnPackOp::generateScalarImplementation(OpBuilder &builder,
                                                     Location loc,
                                                     ValueRange ivs) {
  assert(ivs.size() == getOutputRank() &&
         "number of ivs must match the rank of the output tensor");
  OpBuilder::InsertionGuard g(builder);
  ReifiedRankedShapedTypeDims outputShape;
  if (failed(reifyResultShapes(builder, outputShape))) {
    return getOperation()->emitOpError("failed to reify result shape");
  }
  if (outputShape.size() != 1 || outputShape[0].size() != getOutputRank()) {
    return getOperation()->emitOpError(
               "expected shape of one result value of rank")
           << getOutputRank();
  }

  DenseMap<int64_t, OpFoldResult> dimAndTileMapping = getDimAndTileMapping();
  // untiled loops and tile loops induction variables.
  SmallVector<Value> inputIvs;
  // point loops induction variables.
  SmallVector<Value> inputIvsPointLoops;
  inputIvs.reserve(getOutputRank());
  inputIvsPointLoops.reserve(dimAndTileMapping.size());
  for (auto dim : llvm::seq<int64_t>(0, getOutputRank())) {
    if (dimAndTileMapping.count(dim)) {
      affine::DivModValue divMod =
          affine::getDivMod(builder, loc, ivs[dim],
                            getValueOrCreateConstantIndexOp(
                                builder, loc, dimAndTileMapping[dim]));
      inputIvsPointLoops.push_back(divMod.remainder);
      inputIvs.push_back(divMod.quotient);
    } else {
      inputIvs.push_back(ivs[dim]);
    }
  }

  // TODO: (lorenzo) simplify the logic a bit. There is `ivs`,
  // `inputIvsPointLoops` and `inputIvs`.
  assert(inputIvsPointLoops.size() + inputIvs.size() == getInputRank() &&
         "expect same number of iduction variables equals to input rank");
  // interchange the point loops induction variables based on `inner_dim_pos`.
  ArrayRef<int64_t> innerDims = getInnerDimsPos();
  SmallVector<int64_t> interchangeVector =
      computeInterchangeFromDimPos(innerDims, getOutputRank());
  SmallVector<Value> interchangedInputIvsPointLoops = inputIvsPointLoops;
  interchangedInputIvsPointLoops = interchange<Value>(
      interchangedInputIvsPointLoops, interchangeVector, /*offset=*/0);
  // interchange the tiled loops induction variables based on `outer_dims_perm`.
  ArrayRef<int64_t> outerDims = getOuterDimsPerm();
  if (!outerDims.empty()) {
    inputIvs = interchange<Value>(inputIvs, outerDims, /*offset=*/0);
  }

  llvm::append_range(inputIvs, interchangedInputIvsPointLoops);
  Value scalar = builder.create<memref::LoadOp>(loc, getInput(), inputIvs);
  builder.create<memref::StoreOp>(loc, scalar, getOutput(), ivs);
  return success();
}

SmallVector<Range> UnPackOp::getIterationDomain(OpBuilder &builder) {
  return LinalgExt::getIterationDomain(*this, builder);
}

//===----------------------------------------------------------------------===//
// WinogradInputTransformOp
//===----------------------------------------------------------------------===//

SmallVector<Range>
WinogradInputTransformOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Range> loopBounds(getIterationDomainRank());
  for (int dim = 0; dim < getOutputRank(); ++dim) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = getDimValue(builder, loc, getOutput(), dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

SmallVector<utils::IteratorType>
WinogradInputTransformOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getIterationDomainRank(),
                                                 utils::IteratorType::parallel);
  return iteratorTypes;
}

FailureOr<TilingResult>
WinogradInputTransformOp::getTiledImplementation(OpBuilder &builder,
                                                 ArrayRef<OpFoldResult> offsets,
                                                 ArrayRef<OpFoldResult> sizes) {
  Location loc = getLoc();
  auto zero = builder.getIndexAttr(0);
  const int cDim = getChannelDim();

  assert(offsets.size() == 6);
  SmallVector<int64_t> perm(getInputTileDimensions());
  perm.append(getNonInputTileDims());
  SmallVector<OpFoldResult> offsetsPermuted = applyPermutation(offsets, perm);
  SmallVector<OpFoldResult> sizesPermuted = applyPermutation(sizes, perm);
  SmallVector<OpFoldResult> inputOffsets(getInputRank(), zero);
  const auto hDim = getImageDimensions()[0];
  const auto wDim = getImageDimensions()[1];
  inputOffsets[0] = offsetsPermuted[2];
  inputOffsets[cDim] = offsetsPermuted[5];

  ReifiedRankedShapedTypeDims reifiedInputShapes;
  if (failed(getStaticOrReifiedInputDims(builder, loc, getInput(),
                                         reifiedInputShapes))) {
    return failure();
  }
  SmallVector<OpFoldResult> inputSizes = reifiedInputShapes[0];

  assert(sizes.size() == 6);
  inputSizes[0] = sizesPermuted[2];
  inputSizes[cDim] = sizesPermuted[5];

  auto hSizeAndOffset = getScaledSizeAndOffset(
      builder, loc, sizesPermuted[3], offsetsPermuted[3], inputSizes[hDim],
      getOutputTileSize(), getInputTileSize());
  auto wSizeAndOffset = getScaledSizeAndOffset(
      builder, loc, sizesPermuted[4], offsetsPermuted[4], inputSizes[wDim],
      getOutputTileSize(), getInputTileSize());

  inputSizes[hDim] = hSizeAndOffset.first;
  inputSizes[wDim] = wSizeAndOffset.first;
  inputOffsets[hDim] = hSizeAndOffset.second;
  inputOffsets[wDim] = wSizeAndOffset.second;

  SmallVector<Value> tiledOperands;
  auto one = builder.getIndexAttr(1);
  SmallVector<OpFoldResult> inputStrides(getInputRank(), one);
  tiledOperands.emplace_back(getSlice(builder, loc, getInput(), inputOffsets,
                                      inputSizes, inputStrides));
  SmallVector<OpFoldResult> outputStrides(getOutputRank(), one);
  tiledOperands.emplace_back(
      getSlice(builder, loc, getOutput(), offsets, sizes, outputStrides));

  SmallVector<Type, 4> resultTypes;
  if (hasPureTensorSemantics()) {
    resultTypes.push_back(tiledOperands[1].getType());
  }

  Operation *tiledOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
}

LogicalResult WinogradInputTransformOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber == 0) {
    resultSizes = SmallVector<OpFoldResult>(sizes);
    resultOffsets = SmallVector<OpFoldResult>(offsets);
    return success();
  }
  return failure();
}

FailureOr<TilingResult> WinogradInputTransformOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  int64_t numLoops = getIterationDomainRank();
  return getTiledImplementation(b, offsets.take_front(numLoops),
                                sizes.take_front(numLoops));
}

//===----------------------------------------------------------------------===//
// WinogradFilterTransformOp
//===----------------------------------------------------------------------===//

SmallVector<Range>
WinogradFilterTransformOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  OpFoldResult zero = builder.getIndexAttr(0);
  OpFoldResult one = builder.getIndexAttr(1);
  SmallVector<Range> loopBounds(getOutputRank());
  for (int dim = 0; dim < getOutputRank(); ++dim) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = getDimValue(builder, loc, getOutput(), dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

SmallVector<utils::IteratorType>
WinogradFilterTransformOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getIterationDomainRank(),
                                                 utils::IteratorType::parallel);
  return iteratorTypes;
}

FailureOr<TilingResult> WinogradFilterTransformOp::getTiledImplementation(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  Location loc = getLoc();
  OpFoldResult one = builder.getIndexAttr(1);
  OpFoldResult zero = builder.getIndexAttr(0);
  const int cDim = getChannelDim();
  const int fDim = getFilterDim();

  assert(offsets.size() == 4);
  SmallVector<int64_t> perm(getInputTileDimensions());
  perm.append(getNonInputTileDims());
  SmallVector<OpFoldResult> offsetsPermuted = applyPermutation(offsets, perm);
  SmallVector<OpFoldResult> sizesPermuted = applyPermutation(sizes, perm);
  SmallVector<OpFoldResult> inputOffsets(getInputRank(), zero);
  inputOffsets[cDim] = offsetsPermuted[2];
  inputOffsets[fDim] = offsetsPermuted[3];

  assert(sizes.size() == 4);
  ArrayRef<int64_t> inputShape = getInputType().getShape();
  SmallVector<OpFoldResult> inputSizes =
      getAsOpFoldResult(builder.getIndexArrayAttr(inputShape));
  inputSizes[cDim] = sizesPermuted[2];
  inputSizes[fDim] = sizesPermuted[3];

  SmallVector<Value> tiledOperands;
  SmallVector<OpFoldResult> inputStrides(getInputRank(), one);
  tiledOperands.emplace_back(getSlice(builder, loc, getInput(), inputOffsets,
                                      inputSizes, inputStrides));
  SmallVector<OpFoldResult> outputStrides(getOutputRank(), one);
  tiledOperands.emplace_back(
      getSlice(builder, loc, getOutput(), offsets, sizes, outputStrides));

  SmallVector<Type> resultTypes;
  if (hasPureTensorSemantics()) {
    resultTypes.push_back(tiledOperands[1].getType());
  }

  Operation *tiledOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
}

LogicalResult WinogradFilterTransformOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber != 0) {
    return failure();
  }
  resultSizes = SmallVector<OpFoldResult>(sizes);
  resultOffsets = SmallVector<OpFoldResult>(offsets);
  return success();
}

FailureOr<TilingResult> WinogradFilterTransformOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  int64_t numLoops = getIterationDomainRank();
  return getTiledImplementation(b, offsets.take_front(numLoops),
                                sizes.take_front(numLoops));
}

//===----------------------------------------------------------------------===//
// WinogradOutputTransformOp
//===----------------------------------------------------------------------===//

SmallVector<Range>
WinogradOutputTransformOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Range> loopBounds(getIterationDomainRank());
  for (int dim = 0; dim < getInputRank(); ++dim) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = getDimValue(builder, loc, getInput(), dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

SmallVector<utils::IteratorType>
WinogradOutputTransformOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getIterationDomainRank(),
                                                 utils::IteratorType::parallel);
  return iteratorTypes;
}

FailureOr<TilingResult> WinogradOutputTransformOp::getTiledImplementation(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  Location loc = getLoc();
  auto one = builder.getIndexAttr(1);
  auto zero = builder.getIndexAttr(0);
  const int cDim = getChannelDim();

  assert(offsets.size() == 6);
  SmallVector<int64_t> perm(getInputTileDimensions());
  perm.append(getNonInputTileDims());
  SmallVector<OpFoldResult> sizesPermuted = applyPermutation(sizes, perm);
  SmallVector<OpFoldResult> offsetsPermuted = applyPermutation(offsets, perm);
  const auto hDim = getImageDimensions()[0];
  const auto wDim = getImageDimensions()[1];
  SmallVector<OpFoldResult> outputOffsets(getOutputRank(), zero);

  outputOffsets[0] = offsetsPermuted[2];
  outputOffsets[cDim] = offsetsPermuted[5];

  ReifiedRankedShapedTypeDims reifiedResultShapes;
  if (failed(reifyResultShapes(builder, reifiedResultShapes))) {
    return failure();
  }
  SmallVector<OpFoldResult> outputSizes = reifiedResultShapes[0];

  assert(sizes.size() == 6);
  outputSizes[0] = sizesPermuted[2];
  outputSizes[cDim] = sizesPermuted[5];

  auto hSizeAndOffset = getScaledSizeAndOffset(
      builder, loc, sizesPermuted[3], offsetsPermuted[3], outputSizes[hDim],
      getOutputTileSize(), getOutputTileSize());
  auto wSizeAndOffset = getScaledSizeAndOffset(
      builder, loc, sizesPermuted[4], offsetsPermuted[4], outputSizes[wDim],
      getOutputTileSize(), getOutputTileSize());

  outputSizes[hDim] = hSizeAndOffset.first;
  outputSizes[wDim] = wSizeAndOffset.first;
  outputOffsets[hDim] = hSizeAndOffset.second;
  outputOffsets[wDim] = wSizeAndOffset.second;

  SmallVector<OpFoldResult> outputStrides(getOutputRank(), one);
  Value outputSlice = getSlice(builder, loc, getOutput(), outputOffsets,
                               outputSizes, outputStrides);
  // The image dims of the winograd.output_transform result will always be a
  // multiple of the static output_tile_size, so insert a tensor.cast op to
  // maintain more static information in the IR.
  auto outSliceType = cast<ShapedType>(outputSlice.getType());
  SmallVector<int64_t> staticOutShape(outSliceType.getShape());
  auto constSizeH = getConstantIntValue(sizesPermuted[3]);
  if (constSizeH.has_value()) {
    staticOutShape[hDim] = constSizeH.value() * getOutputTileSize();
  }
  auto constSizeW = getConstantIntValue(sizesPermuted[4]);
  if (constSizeW.has_value()) {
    staticOutShape[wDim] = constSizeW.value() * getOutputTileSize();
  }
  Value staticOutputSlice =
      castValue(builder, loc, outputSlice, outSliceType.clone(staticOutShape));

  SmallVector<Value> tiledOperands;
  SmallVector<OpFoldResult> inputStrides(getInputRank(), one);
  tiledOperands.emplace_back(
      getSlice(builder, loc, getInput(), offsets, sizes, inputStrides));
  tiledOperands.emplace_back(staticOutputSlice);

  SmallVector<Type, 4> resultTypes;
  if (hasPureTensorSemantics()) {
    resultTypes.push_back(tiledOperands[1].getType());
  }

  Operation *tiledOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  SmallVector<Value> results(tiledOp->getResults());
  if (!results.empty()) {
    results.front() = castValue(builder, loc, results.front(), outSliceType);
  }
  return TilingResult{{tiledOp}, results};
}

LogicalResult WinogradOutputTransformOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber == 0) {
    auto resultShape = cast<ShapedType>(getOutput().getType()).getShape();
    resultSizes = getAsOpFoldResult(builder.getIndexArrayAttr(resultShape));
    resultOffsets =
        SmallVector<OpFoldResult>(getOutputRank(), builder.getIndexAttr(0));
    const int cDim = getChannelDim();
    const auto hDim = getImageDimensions()[0];
    const auto wDim = getImageDimensions()[1];
    SmallVector<int64_t> perm(getInputTileDimensions());
    perm.append(getNonInputTileDims());
    SmallVector<OpFoldResult> sizesPermuted = applyPermutation(sizes, perm);
    SmallVector<OpFoldResult> offsetsPermuted = applyPermutation(offsets, perm);
    auto loc = getLoc();
    resultOffsets[0] = offsetsPermuted[2];
    resultOffsets[cDim] = offsetsPermuted[5];
    resultSizes[0] = sizesPermuted[2];
    resultSizes[cDim] = sizesPermuted[5];
    SmallVector<SmallVector<OpFoldResult>> reifiedResultShapes;
    if (failed(reifyResultShapes(builder, reifiedResultShapes))) {
      return failure();
    }
    auto hSizeAndOffset = getScaledSizeAndOffset(
        builder, loc, sizesPermuted[3], offsetsPermuted[3],
        reifiedResultShapes[0][hDim], getOutputTileSize(), getOutputTileSize());
    auto wSizeAndOffset = getScaledSizeAndOffset(
        builder, loc, sizesPermuted[4], offsetsPermuted[4],
        reifiedResultShapes[0][wDim], getOutputTileSize(), getOutputTileSize());

    resultSizes[hDim] = hSizeAndOffset.first;
    resultSizes[wDim] = wSizeAndOffset.first;
    resultOffsets[hDim] = hSizeAndOffset.second;
    resultOffsets[wDim] = wSizeAndOffset.second;
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// AttentionOp
//===----------------------------------------------------------------------===//

SmallVector<Range> AttentionOp::getIterationDomain(OpBuilder &builder) {
  int64_t iterationDomainRank = getIterationDomainRank();
  SmallVector<Range> loopBounds(iterationDomainRank);
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = getQuery();
  for (auto dim : llvm::seq<int64_t>(0, iterationDomainRank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = getDimValue(builder, loc, source, dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

SmallVector<utils::IteratorType> AttentionOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getIterationDomainRank(),
                                                 utils::IteratorType::parallel);
  return iteratorTypes;
}

FailureOr<TilingResult>
AttentionOp::getTiledImplementation(OpBuilder &builder,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> sizes) {
  assert(offsets.size() == getIterationDomainRank());
  assert(sizes.size() == getIterationDomainRank());

  Location loc = getLoc();
  auto one = builder.getIndexAttr(1);
  auto zero = builder.getIndexAttr(0);

  SmallVector<OpFoldResult> queryOutputOffsets(getQueryRank(), zero);
  SmallVector<OpFoldResult> queryOutputStrides(getQueryRank(), one);
  ArrayRef<int64_t> queryShape = getQueryType().getShape();
  SmallVector<OpFoldResult> queryOutputSizes =
      getAsOpFoldResult(builder.getIndexArrayAttr(queryShape));
  for (auto [idx, info] : llvm::enumerate(llvm::zip_equal(offsets, sizes))) {
    queryOutputOffsets[idx] = std::get<0>(info);
    queryOutputSizes[idx] = std::get<1>(info);
  }

  SmallVector<OpFoldResult> keyValueOffsets(getKeyRank(), zero);
  SmallVector<OpFoldResult> keyValueStrides(getKeyRank(), one);
  ArrayRef<int64_t> keyShape = getKeyType().getShape();
  SmallVector<OpFoldResult> keyValueSizes =
      getAsOpFoldResult(builder.getIndexArrayAttr(keyShape));
  keyValueSizes[0] = sizes[0];
  keyValueOffsets[0] = offsets[0];

  Value scale = getScale();

  SmallVector<Value> tiledOperands;
  tiledOperands.emplace_back(getSlice(builder, loc, getQuery(),
                                      queryOutputOffsets, queryOutputSizes,
                                      queryOutputStrides));
  tiledOperands.emplace_back(getSlice(builder, loc, getKey(), keyValueOffsets,
                                      keyValueSizes, keyValueStrides));
  tiledOperands.emplace_back(getSlice(builder, loc, getValue(), keyValueOffsets,
                                      keyValueSizes, keyValueStrides));
  tiledOperands.emplace_back(scale);
  tiledOperands.emplace_back(getSlice(builder, loc, getOutput(),
                                      queryOutputOffsets, queryOutputSizes,
                                      queryOutputStrides));

  SmallVector<Type> resultTypes;
  if (hasPureTensorSemantics())
    resultTypes.push_back(tiledOperands[4].getType());

  Operation *tiledOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
}

LogicalResult AttentionOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber == 0) {
    ArrayRef<int64_t> resultShape = getOutputType().getShape();
    resultSizes = getAsOpFoldResult(builder.getIndexArrayAttr(resultShape));
    resultOffsets =
        SmallVector<OpFoldResult>(getOutputRank(), builder.getIndexAttr(0));
    for (auto [idx, info] : llvm::enumerate(llvm::zip_equal(offsets, sizes))) {
      resultOffsets[idx] = std::get<0>(info);
      resultSizes[idx] = std::get<1>(info);
    }
    return success();
  }
  return failure();
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
