// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-generic-vectorization"
#define VEC_DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define VEC_LDBG(X) LLVM_DEBUG(VEC_DBGS() << X << "\n")

namespace mlir::iree_compiler {

namespace {

struct VectorizationTileSizes {
  SmallVector<int64_t> destShape;
  SmallVector<int64_t> vectorSizes;
};

/// Returns a VectorizationTileSizes which contains the inferred bounded result
/// shape and vector input sizes. This is useful to infer the sizes from a
/// chain.
static std::optional<VectorizationTileSizes> inferSizesFromIR(Value val);

/// Tries to infer the vector sizes from an IR using ValueBounds analysis. If
/// `opResult` is provided, it stores the bounded result shapes to destShape.
/// Returns std::nullopt if vector sizes can't be inferred.
static std::optional<VectorizationTileSizes>
inferSizesFromIR(linalg::LinalgOp linalgOp, std::optional<OpResult> opResult) {
  LLVM_DEBUG(VEC_DBGS() << "Inferring sizes for:\n"
                        << linalgOp << " with OpResult.resultNumber="
                        << opResult->getResultNumber() << "\n");

  VectorizationTileSizes result;
  unsigned numDims = linalgOp.getNumLoops();

  for (int dim = 0; dim < numDims; ++dim) {
    // Map dimension `dim` to an operand dimension that we will use to
    // traverse the U-D chain to get `dim` vector size information.
    SmallVector<std::pair<Value, unsigned>> operandDimPairs;
    linalgOp.mapIterationSpaceDimToAllOperandDims(dim, operandDimPairs);
    if (operandDimPairs.empty()) {
      return std::nullopt;
    }

    Value firstOperand = operandDimPairs[0].first;
    unsigned firstOperandDim = operandDimPairs[0].second;

    // Trivial case: `dim` size is available in the operand type.
    int64_t dimSize = llvm::cast<ShapedType>(firstOperand.getType())
                          .getShape()[firstOperandDim];
    if (!ShapedType::isDynamic(dimSize)) {
      result.vectorSizes.push_back(dimSize);
      LLVM_DEBUG(VEC_DBGS() << "Inferred iteration size '" << dimSize
                            << "' for dimension '" << dim << "'\n");
      continue;
    }

    // Use ValueBounds analysis to infer `dim` size upper bound.
    FailureOr<int64_t> maybeDimBound;
    for (auto operandDimPair : operandDimPairs) {
      Value operand = operandDimPair.first;
      unsigned operandDim = operandDimPair.second;
      maybeDimBound = ValueBoundsConstraintSet::computeConstantBound(
          presburger::BoundType::UB, operand, operandDim,
          /*stopCondition=*/nullptr, /*closedUB=*/true);

      if (succeeded(maybeDimBound)) {
        break;
      }
    }

    if (failed(maybeDimBound)) {
      return std::nullopt;
    }

    dimSize = maybeDimBound.value();
    result.vectorSizes.push_back(dimSize);
    LLVM_DEBUG(VEC_DBGS() << "Inferred iteration size '" << dimSize
                          << "' for dimension '" << dim << "'\n");
  }

  if (opResult) {
    result.destShape = linalgOp.getIndexingMapMatchingResult(opResult.value())
                           .compose(result.vectorSizes);
  }

  return result;
}

/// Returns the result sizes and vector input sizes of the tensor.pack op. The
/// inferred bounding size is returned if it is dynamic shape. Returns
/// std::nullopt if the shape inference failed.
static std::optional<VectorizationTileSizes>
inferSizesFromIR(tensor::PackOp op) {
  LLVM_DEBUG(VEC_DBGS() << "Inferring dest sizes for:\n" << op << "\n");

  if (llvm::any_of(op.getInnerTiles(), [](OpFoldResult v) {
        return !getConstantIntValue(v).has_value();
      })) {
    LLVM_DEBUG(VEC_DBGS() << "skip, because inner_tiles are not all constant");
    return std::nullopt;
  }

  VectorizationTileSizes result;
  std::optional<VectorizationTileSizes> inferred =
      inferSizesFromIR(op.getSource());
  if (!inferred) {
    return std::nullopt;
  }
  result.vectorSizes = inferred.value().destShape;

  for (auto [dimPos, tileSize] :
       llvm::zip_equal(op.getInnerDimsPos(), op.getStaticInnerTiles())) {
    if (result.vectorSizes[dimPos] % tileSize != 0) {
      return std::nullopt;
    }
    result.vectorSizes[dimPos] /= tileSize;
  }
  auto outerDimsPerm = op.getOuterDimsPerm();
  if (!outerDimsPerm.empty()) {
    applyPermutationToVector(result.vectorSizes, outerDimsPerm);
  }

  LLVM_DEBUG({
    VEC_DBGS() << "After adjustment with inner tiles and "
                  "outer_dims_perm:\n";
    for (auto [idx, val] : llvm::enumerate(result.vectorSizes)) {
      llvm::dbgs() << "Dim #" << idx << ": " << val << "\n";
    }
  });
  result.destShape = result.vectorSizes;

  return result;
}

/// Returns the result sizes and vector input sizes of the tensor.unpack op. The
/// inferred bounding size is returned if it is dynamic shape. Returns
/// std::nullopt if the shape inference failed.
static std::optional<VectorizationTileSizes>
inferSizesFromIR(tensor::UnPackOp op) {
  LLVM_DEBUG(VEC_DBGS() << "Inferring dest sizes for:\n" << op << "\n");

  if (llvm::any_of(op.getInnerTiles(), [](OpFoldResult v) {
        return !getConstantIntValue(v).has_value();
      })) {
    LLVM_DEBUG(
        VEC_DBGS()
        << "failed on inference because inner_tiles are not all constant");
    return std::nullopt;
  }

  VectorizationTileSizes result;
  std::optional<VectorizationTileSizes> inferred =
      inferSizesFromIR(op.getSource());
  if (!inferred) {
    return std::nullopt;
  }
  result.vectorSizes = inferred.value().destShape;

  result.vectorSizes.resize(op.getDestType().getRank());
  auto outerDimsPerm = op.getOuterDimsPerm();
  if (!outerDimsPerm.empty()) {
    applyPermutationToVector(result.vectorSizes,
                             invertPermutationVector(outerDimsPerm));
  }
  for (auto [dimPos, tileSize] :
       llvm::zip_equal(op.getInnerDimsPos(), op.getStaticInnerTiles())) {
    result.vectorSizes[dimPos] *= tileSize;
  }

  LLVM_DEBUG({
    VEC_DBGS() << "After adjustment with inner tiles and "
                  "outer_dims_perm:\n";
    for (auto [idx, val] : llvm::enumerate(result.vectorSizes)) {
      llvm::dbgs() << "Dim #" << idx << ": " << val << "\n";
    }
  });
  result.destShape = result.vectorSizes;

  return result;
}

/// See the documentation in the above function declaration.
static std::optional<VectorizationTileSizes> inferSizesFromIR(Value val) {
  std::optional<VectorizationTileSizes> result;
  TypeSwitch<Operation *, void>(val.getDefiningOp())
      .Case<linalg::LinalgOp>(
          [&](auto op) { result = inferSizesFromIR(op, cast<OpResult>(val)); })
      .Case<tensor::PackOp>([&](auto op) { result = inferSizesFromIR(op); })
      .Case<tensor::ExtractSliceOp>([&](tensor::ExtractSliceOp op) {
        // tensor::ExtractSliceOp is not vectorizable, so only `destShape` has
        // the values.
        result = VectorizationTileSizes();
        LLVM_DEBUG(VEC_DBGS() << "Inferring sizes for:\n" << op << "\n");
        int64_t destRank = op.getResult().getType().getRank();
        for (int dim = 0; dim < destRank; ++dim) {
          LLVM_DEBUG(VEC_DBGS() << "Dim #" << dim << ": ");
          FailureOr<int64_t> maybeDimBound =
              ValueBoundsConstraintSet::computeConstantBound(
                  presburger::BoundType::UB, op, dim,
                  /*stopCondition=*/nullptr, /*closedUB=*/true);
          if (failed(maybeDimBound)) {
            LLVM_DEBUG(llvm::dbgs() << "failed\n");
            result = std::nullopt;
            return;
          }
          LLVM_DEBUG(llvm::dbgs() << maybeDimBound.value() << "\n");
          result->destShape.push_back(maybeDimBound.value());
        }
      })
      .Default([&](Operation *) {});
  return result;
}

// Returns the vector sizes from the local lowering config or try to infer them
// from the tensor shapes and tiled loops in the IR.
static std::optional<SizesAndScalableFlags>
getVectorSizes(Operation *op, bool useConfiguredVectorSizes) {
  // Get vector sizes from the lowering config, if available in the op itself.
  IREE::Codegen::LoweringConfigAttr loweringConfig = getLoweringConfig(op);
  if (useConfiguredVectorSizes && loweringConfig) {
    TilingConfig tilingConfig(loweringConfig);
    auto [vectorSizes, scalableFlags] = tilingConfig.getVectorTileSizes();
    // Replace zeros in canonical vector shape to turn it into a valid shape.
    std::replace(vectorSizes.begin(), vectorSizes.end(), 0, 1);
    return std::make_pair(vectorSizes, scalableFlags);
  }

  // Try to infer the vector sizes from the IR.
  std::optional<SmallVector<int64_t>> vectorSizes;
  TypeSwitch<Operation *, void>(op)
      .Case<linalg::LinalgOp>([&](linalg::LinalgOp linalgOp) {
        std::optional<VectorizationTileSizes> result =
            inferSizesFromIR(linalgOp, /*opResult=*/std::nullopt);
        if (result) {
          vectorSizes = result->vectorSizes;
        }
      })
      .Case<tensor::PackOp, tensor::UnPackOp>([&](auto op) {
        std::optional<VectorizationTileSizes> result = inferSizesFromIR(op);
        if (result) {
          vectorSizes = result->vectorSizes;
        }
      })
      .Case<tensor::PadOp>([&](tensor::PadOp padOp) {
        auto ty = padOp.getResultType();
        // TODO(hanchung): Infer the vector sizes for pad op after
        // maskedVectorize method allows dynamic result shapes.
        if (!ty.hasStaticShape())
          return;
        vectorSizes = SmallVector<int64_t>(ty.getShape());
      })
      .Default([&](Operation *) {});

  if (vectorSizes) {
    // This can't identify scalable flags, so pad them with `false`.
    return std::make_pair(vectorSizes.value(),
                          SmallVector<bool>(vectorSizes->size(), false));
  }
  return std::nullopt;
}

static LogicalResult isWithinVectorSizeLimit(linalg::LinalgOp linalgOp,
                                             int64_t maxVectorSize) {
  int64_t maxFlatVecSize = 1;
  for (OpOperand &operand : linalgOp->getOpOperands()) {
    auto type = llvm::dyn_cast<ShapedType>(operand.get().getType());
    if (!type)
      continue;
    if (!type.hasStaticShape())
      return failure();
    maxFlatVecSize = std::max(maxFlatVecSize, type.getNumElements());
  }
  return success(maxFlatVecSize < maxVectorSize);
}

class GenericVectorizationPass
    : public GenericVectorizationBase<GenericVectorizationPass> {
public:
  using GenericVectorizationBase::GenericVectorizationBase;
  GenericVectorizationPass(const GenericVectorizationPassOptions &options) {
    this->enableVectorMasking.setValue(options.enableVectorMasking);
    this->useConfiguredVectorSizes.setValue(options.useConfiguredVectorSizes);
    this->vectorizePadding.setValue(options.vectorizePadding);
    this->vectorizeGatherAccesses.setValue(options.vectorizeGatherAccesses);
    this->enableCleanup.setValue(options.enableCleanup);
    this->generateContract.setValue(options.generateContract);
    this->foldCastIntoContract.setValue(options.foldCastIntoContract);
    this->maxVectorSize.setValue(options.maxVectorSize);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override;
};

/// Returns success if `inputVectorSizes` is a valid masking configuraion for
/// given `shape`, i.e., it meets:
///   1. The numbers of elements in both array are equal.
///   2. `inputVectorSizes` does nos have dynamic dimensions.
///   3. All the values in `inputVectorSizes` are greater than or equal to
///      static sizes in `shape`.
static LogicalResult
isValidMaskedInputVector(ArrayRef<int64_t> shape,
                         ArrayRef<int64_t> inputVectorSizes) {
  VEC_LDBG("Iteration space static sizes:");
  LLVM_DEBUG(llvm::interleaveComma(shape, llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  if (inputVectorSizes.size() != shape.size()) {
    VEC_LDBG("Input vector sizes don't match the number of loops");
    return failure();
  }
  if (ShapedType::isDynamicShape(inputVectorSizes)) {
    VEC_LDBG("Input vector sizes can't have dynamic dimensions");
    return failure();
  }
  if (!llvm::all_of(llvm::zip(shape, inputVectorSizes),
                    [](std::tuple<int64_t, int64_t> sizePair) {
                      int64_t staticSize = std::get<0>(sizePair);
                      int64_t inputSize = std::get<1>(sizePair);
                      return ShapedType::isDynamic(staticSize) ||
                             staticSize <= inputSize;
                    })) {
    VEC_LDBG(
        "Input vector sizes must be greater than or equal to iteration space "
        "static sizes");
    return failure();
  }
  return success();
}

static LogicalResult
vectorizeTopkOpPrecondition(IREE::LinalgExt::TopkOp topkOp,
                            ArrayRef<int64_t> inputVectorSizes) {

  auto out0 = topkOp.getResults()[0];
  auto resShapedTy = llvm::cast<ShapedType>(out0.getType());
  ArrayRef<int64_t> resShape = resShapedTy.getShape();
  if (resShapedTy.isDynamicShape(resShape))
    return failure();

  auto inShapedTy = topkOp.getInputType();
  ArrayRef<int64_t> inShape = inShapedTy.getShape();

  // Validate input
  auto inElemTy = topkOp.getInputType().getElementType();
  if (auto intType = llvm::dyn_cast_if_present<IntegerType>(inElemTy)) {
    // Do nothing. Expected type.
  } else if (auto floatType = llvm::dyn_cast_if_present<FloatType>(inElemTy)) {
    // Do nothing. Expected type.
  } else {
    // Unexpected type.
    return failure();
  }

  if (failed(isValidMaskedInputVector(inShape.take_front(topkOp.getInputRank()),
                                      inputVectorSizes)))
    return failure();
  return success();
}

static scf::ForOp replaceForOpWithNewSignature(RewriterBase &rewriter,
                                               scf::ForOp loop,
                                               ValueRange newInitArgs) {
  OpBuilder::InsertionGuard g(rewriter);
  // Create a new loop before the existing one, with the extra operands.
  rewriter.setInsertionPoint(loop);
  auto operands = llvm::to_vector<4>(loop.getInitArgs());
  llvm::append_range(operands, newInitArgs);
  scf::ForOp newLoop = rewriter.create<scf::ForOp>(
      loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
      operands);
  rewriter.eraseBlock(newLoop.getBody());

  newLoop.getRegion().getBlocks().splice(
      newLoop.getRegion().getBlocks().begin(), loop.getRegion().getBlocks());

  newLoop.getBody()->getTerminator()->print(llvm::errs());

  // newLoop.getRegion().print(llvm::errs());
  for (Value operand : newInitArgs) {
    newLoop.getBody()->addArgument(operand.getType(), operand.getLoc());
  }

  for (auto it : llvm::zip(loop.getResults(), newLoop.getResults().take_front(
                                                  loop.getNumResults())))
    rewriter.replaceAllUsesWith(std::get<0>(it), std::get<1>(it));

  LLVM_DEBUG(VEC_DBGS() << "newLoop now: " << newLoop << "\n");
  LLVM_DEBUG(VEC_DBGS() << "stripped scf.for: " << loop << "\n");
  LLVM_DEBUG(VEC_DBGS() << "erase: " << loop);

  rewriter.eraseOp(loop);
  return newLoop;
}

/// Add the necessary IterArgs to the input iterating loop.
static LogicalResult addIterArgs(RewriterBase &rewriter, Location loc,
                                 scf::ForOp loop, Type outValType,
                                 Type outIdxType) {
  llvm::DenseMap<Value, Value> valueMapping;
  SmallVector<Value> newOperands;
  SmallVector<std::pair<size_t, size_t>> argMapping;
  // First, copy the existing loop args.
  for (const auto &operand : llvm::enumerate(loop.getInitArgs())) {
    auto it = valueMapping.find(operand.value());
    if (it == valueMapping.end()) {
      LLVM_DEBUG(VEC_DBGS()
                 << "no value mapping for: " << operand.value() << "\n");
      continue;
    }

    argMapping.push_back(std::make_pair(
        operand.index(), loop.getInitArgs().size() + newOperands.size()));
    newOperands.push_back(it->second);
  }

  // // Add an arg wheter smallestElem was initialized
  Value firstElemInit = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
  newOperands.push_back(firstElemInit);

  // Add an arg for the smallest element added to the out array
  Value smallestOut;
  if (auto floatType = llvm::dyn_cast_if_present<FloatType>(outValType)) {
    smallestOut = rewriter.create<arith::ConstantFloatOp>(
        loc, mlir::APFloat(floatType.getFloatSemantics(), 0), floatType);
  } else if (auto intType =
                 llvm::dyn_cast_if_present<IntegerType>(outValType)) {
    smallestOut = rewriter.create<arith::ConstantIntOp>(loc, 0, intType);
  } else {
    // Unexpected type!
    return failure();
  }
  newOperands.push_back(smallestOut);

  // Add an arg for the number of elements added to the out array(s).
  Value numElemsAdded = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  newOperands.push_back(numElemsAdded);
  scf::ForOp newForOp =
      replaceForOpWithNewSignature(rewriter, loop, newOperands);
  Block &loopBody = *newForOp.getBody();
  for (auto mapping : argMapping) {
    valueMapping[newForOp.getResult(mapping.first)] =
        newForOp.getResult(mapping.second);
    valueMapping[loopBody.getArgument(mapping.first +
                                      newForOp.getNumInductionVars())] =
        loopBody.getArgument(mapping.second + newForOp.getNumInductionVars());
  }
  return success();
}

/// Create a TransferReadOp from `source` with static shape `readShape`. If the
/// vector type for the read is not the same as the type of `source`, then a
/// mask is created on the read.
static Value createReadOrMaskedRead(RewriterBase &rewriter, Location loc,
                                    Value source, ArrayRef<int64_t> readShape,
                                    Type elemType) {
  assert(llvm::none_of(readShape,
                       [](int64_t s) { return s == ShapedType::kDynamic; }));
  Value padValue = rewriter.create<arith::ConstantOp>(
      loc, elemType, rewriter.getZeroAttr(elemType));
  auto sourceShape = dyn_cast<ShapedType>(source.getType()).getShape();
  assert(sourceShape.size() == readShape.size());
  auto maskType = VectorType::get(readShape, rewriter.getI1Type());
  auto vectorType = VectorType::get(readShape, padValue.getType());
  int64_t readRank = readShape.size();
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto transferReadOp = rewriter.create<vector::TransferReadOp>(
      loc,
      /*vectorType=*/vectorType,
      /*source=*/source,
      /*indices=*/SmallVector<Value>(readRank, zero),
      /*padding=*/padValue,
      /*inBounds=*/SmallVector<bool>(readRank, true));
  if (llvm::equal(readShape, sourceShape)) {
    return transferReadOp;
  }
  SmallVector<OpFoldResult> mixedSourceDims =
      tensor::getMixedSizes(rewriter, loc, source);
  Value mask =
      rewriter.create<vector::CreateMaskOp>(loc, maskType, mixedSourceDims);
  return mlir::vector::maskOperation(rewriter, transferReadOp, mask)
      ->getResult(0);
}

struct InsertElemContinuationValues {
  Value out0;         // OutputValues
  Value out1;         // OutputIndices
  Value smallestElem; // Smallest Element
  Value addedElems;   // The number of elements added to the output
};

/// Insert a new, bigger element than the currently smallest element added in
/// output of the function.
///
/// 1. Find the position of the element that needs to be inserted.
///    There is no support for returning in the middle of a loop,
///    so a iter_arg flag is set to annotate that when the insertion
///    index is found.
/// 2. If the insertion index is equal of the dimension of output,
///    no addition is needed.
/// 3. If the insertion index is equal of the elements added to the output,
///    No need to shift, just add the element at the end and expand the
///    addedElems.
/// Returns the out0, out1, smallestElem, addedElems values for continuation.
InsertElemContinuationValues
insertElemInOutput(Location loc, OpBuilder b, Value elemToInsertIfNeeded,
                   Value elemIndex, Value outputNum,
                   InsertElemContinuationValues continuation) {
  SmallVector<Value> newOperands;
  // Flag wheter we have not found the first smaller element.
  newOperands.push_back(b.create<arith::ConstantIntOp>(loc, 1, 1));
  // Keeps the index of where to insert the element that we return.
  newOperands.push_back(b.create<arith::ConstantIndexOp>(loc, 0));

  scf::YieldOp scfYieldOp;
  scf::ForOp findSmaller = b.create<scf::ForOp>(
      loc, b.create<arith::ConstantIndexOp>(loc, 0), continuation.addedElems,
      b.create<arith::ConstantIndexOp>(loc, 1), newOperands,
      [&](OpBuilder &bIn, Location nestedLoc, Value iv, ValueRange args) {
        scfYieldOp = b.create<scf::YieldOp>(nestedLoc, newOperands);
      });

  // Fill in the body for findSmaller Loop
  // We need the iter_args and they are not created yet in the lambda.
  {
    PatternRewriter::InsertionGuard guard(b);
    b.setInsertionPoint(scfYieldOp);
    // Flag wheter we found the insertion point.
    Value notFoundSmaller = findSmaller.getRegionIterArgs()[0];
    // The index where the element needs to be inserted.
    Value insertionIndex = findSmaller.getRegionIterArgs()[1];
    // If we have found the insertionIndex in previous iterations, do nothing.
    Value cmpElemOp = b.create<arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, notFoundSmaller,
        b.create<arith::ConstantIntOp>(loc, 1, 1));
    auto ifFoundIndex = b.create<scf::IfOp>(
        loc, cmpElemOp,
        [&](OpBuilder &b, Location loc) {
          // Not found yet.
          Value notFoundSm = notFoundSmaller;
          Value insertionInd = insertionIndex;

          Value idxDim0 = b.create<arith::ConstantIndexOp>(loc, 0);
          Value inElem = b.create<tensor::ExtractOp>(
              loc, continuation.out0,
              ValueRange{idxDim0, findSmaller.getInductionVar()});
          // First, check if this is the position where the new element needs to
          // be inserted.
          if (auto floatType = llvm::dyn_cast_if_present<FloatType>(
                  elemToInsertIfNeeded.getType())) {
            cmpElemOp =
                b.create<arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT,
                                        inElem, elemToInsertIfNeeded);
          } else if (auto intType = llvm::dyn_cast_if_present<IntegerType>(
                         elemToInsertIfNeeded.getType())) {
            cmpElemOp =
                b.create<arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
                                        inElem, elemToInsertIfNeeded);
          } else {
            assert(false && "Invalid type for topk vectorization!");
          }

          cmpElemOp = b.create<arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::eq, cmpElemOp,
              b.create<arith::ConstantIntOp>(loc, 1, 1));
          auto ifToAdd = b.create<scf::IfOp>(
              loc, cmpElemOp,
              [&](OpBuilder &b, Location loc) {
                // Get the value of the current induction variable.
                Value insertionIndex = b.create<arith::AddIOp>(
                    loc, findSmaller.getInductionVar(),
                    b.create<arith::ConstantIndexOp>(loc, 0));
                SmallVector<Value> newOperands;
                // Set the flag that we have "found" the index where to insert.
                newOperands.push_back(
                    b.create<arith::ConstantIntOp>(loc, 0, 1));
                newOperands.push_back(insertionIndex);
                b.create<scf::YieldOp>(loc, newOperands);
              },
              [&](OpBuilder &b, Location loc) {
                // The index was not found yet.
                Value notFoundSmIn = notFoundSm;
                Value insertionIndIn = insertionInd;
                SmallVector<Value> newOperands;
                newOperands.push_back(notFoundSmIn);
                newOperands.push_back(insertionIndIn);
                b.create<scf::YieldOp>(loc, newOperands);
              });

          SmallVector<Value> newOperands;
          newOperands.push_back(ifToAdd.getResult(0));
          newOperands.push_back(ifToAdd.getResult(1));
          b.create<scf::YieldOp>(loc, newOperands);
        },
        [&](OpBuilder &b, Location loc) {
          // The index was already found yet.
          // Just return. No need to update anything.
          SmallVector<Value> newOperands;
          Value notFoundSm = notFoundSmaller;
          Value insertionInd = insertionIndex;
          newOperands.push_back(notFoundSm);
          newOperands.push_back(insertionInd);
          b.create<scf::YieldOp>(loc, newOperands);
        });

    notFoundSmaller = ifFoundIndex.getResult(0);
    insertionIndex = ifFoundIndex.getResult(1);

    SmallVector<Value> newOperands;
    newOperands.push_back(notFoundSmaller);
    newOperands.push_back(insertionIndex);
    b.create<scf::YieldOp>(loc, newOperands);
    scfYieldOp.erase();
  }

  Value insertionIndexNotFound = findSmaller.getResult(0);
  Value insertionIndex = findSmaller.getResult(1);

  // If the insertionIndex is not found and the addedElems == output dimension,
  // do nothing. The element should not be added.
  Value cmp = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                      continuation.addedElems, outputNum);
  cmp = b.create<arith::AndIOp>(loc, cmp, insertionIndexNotFound);
  auto ifAddElem = b.create<scf::IfOp>(
      loc, cmp,
      [&](OpBuilder &b, Location loc) {
        // if
        // Element doesn't need to be added to the output.
        // It is smaller than the last element in the out dimension.
        SmallVector<Value> newOperands;
        newOperands.push_back(continuation.out0);
        newOperands.push_back(continuation.out1);
        newOperands.push_back(continuation.smallestElem);
        newOperands.push_back(continuation.addedElems);
        b.create<scf::YieldOp>(loc, newOperands);
      },
      [&](OpBuilder &b, Location loc) {
        // else
        // If the insertionIndex not found, element is added at the end.
        // No need to shift.
        // The condition above exclude the case where we have filled the output
        // tensor already.
        auto ifAddAtEnd = b.create<scf::IfOp>(
            loc, insertionIndexNotFound,
            [&](OpBuilder &b, Location loc) {
              // if
              // Adding at addedElems and expanding the addedElems
              Value outValsIf = continuation.out0;
              Value outIndsIf = continuation.out1;
              ;
              Value smElemIf = continuation.smallestElem;
              Value addElemsIf = continuation.addedElems;
              Value idx0 = b.create<arith::ConstantIndexOp>(loc, 0);

              outValsIf = b.create<tensor::InsertOp>(
                  loc, elemToInsertIfNeeded, outValsIf,
                  ValueRange{idx0, addElemsIf});
              Value elemIndexI32 =
                  b.create<arith::IndexCastOp>(loc, b.getI32Type(), elemIndex);
              outIndsIf = b.create<tensor::InsertOp>(
                  loc, elemIndexI32, outIndsIf, ValueRange{idx0, addElemsIf});
              auto one = b.create<arith::ConstantIndexOp>(loc, 1);
              addElemsIf = b.create<arith::AddIOp>(loc, addElemsIf, one);
              SmallVector<Value> newOperandsIf;
              newOperandsIf.push_back(outValsIf);
              newOperandsIf.push_back(outIndsIf);
              newOperandsIf.push_back(smElemIf);
              newOperandsIf.push_back(addElemsIf);
              b.create<scf::YieldOp>(loc, newOperandsIf);
            },
            [&](OpBuilder &b, Location loc) {
              Value outValsElse = continuation.out0;
              Value outIndsElse = continuation.out1;
              ;
              Value smElemElse = continuation.smallestElem;
              Value addElemsElse = continuation.addedElems;
              // Get the new end-index for moving the array.
              // It is one less the last index of an added element in the
              // output, because the move is done using a[i+1] = a[i] shifting.
              Value cmp = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  addElemsElse, outputNum);
              auto ifExpand = b.create<scf::IfOp>(
                  loc, cmp,
                  [&](OpBuilder &b, Location loc) {
                    // if
                    // No expansion. Shift elements out of the output.
                    Value addElems = addElemsElse;
                    auto one = b.create<arith::ConstantIndexOp>(loc, 1);
                    Value lastElemIndex =
                        b.create<arith::SubIOp>(loc, addElems, one);
                    SmallVector<Value> newOperands;
                    newOperands.push_back(lastElemIndex);
                    b.create<scf::YieldOp>(loc, newOperands);
                  },
                  [&](OpBuilder &b, Location loc) {
                    // else
                    // Expansion
                    Value addElems = addElemsElse;
                    SmallVector<Value> newOperands;
                    newOperands.push_back(addElems);
                    b.create<scf::YieldOp>(loc, newOperands);
                  });

              Value lastElemIndex = ifExpand.getResult(0);
              // Replacing out[i+1] with out[i].
              // Make sure not over the index.
              auto one = b.create<arith::ConstantIndexOp>(loc, 1);
              // The following loop assumes that the iter_args are
              // the value of the element we are replacing in the next element,
              // for the both out arrays as 0th and 1st RegionIterArg.
              // Note: The elements of the a[i+1] are extracted, then the a[i+1]
              // elements are replaced with a[i] elements, and then the
              // extracted a[i+1] are placed as iter_args 0 and 1.
              Value idx0 = b.create<arith::ConstantIndexOp>(loc, 0);
              Value out1Repl = b.create<tensor::ExtractOp>(
                  loc, outValsElse, ValueRange{idx0, insertionIndex});
              Value out2Repl = b.create<tensor::ExtractOp>(
                  loc, outIndsElse, ValueRange{idx0, insertionIndex});
              SmallVector<Value> newOperands;
              newOperands.push_back(outValsElse);
              newOperands.push_back(outIndsElse);
              newOperands.push_back(out1Repl);
              newOperands.push_back(out2Repl);
              scf::YieldOp lastLoopOp;
              // Now shift the elements
              scf::ForOp shiftElemsLoop = b.create<scf::ForOp>(
                  loc, insertionIndex, lastElemIndex,
                  b.create<arith::ConstantIndexOp>(loc, 1), newOperands,
                  [&](OpBuilder &bIn, Location nestedLoc, Value iv,
                      ValueRange args) {
                    lastLoopOp =
                        bIn.create<scf::YieldOp>(nestedLoc, newOperands);
                  });

              // Create the loop body.
              // Need access to RegionIterArgs, which are not constructed in the
              // lambda yet.
              {
                PatternRewriter::InsertionGuard guard(b);
                // auto ip = rewriter.saveInsertionPoint();
                b.setInsertionPoint(lastLoopOp);
                Value nextIndVar = shiftElemsLoop.getInductionVar();
                auto one = b.create<arith::ConstantIndexOp>(loc, 1);
                nextIndVar = b.create<arith::AddIOp>(loc, nextIndVar, one);
                Value idx0 = b.create<arith::ConstantIndexOp>(loc, 0);
                Value out1Repl = b.create<tensor::ExtractOp>(
                    loc, shiftElemsLoop.getRegionIterArgs()[0],
                    ValueRange{idx0, nextIndVar});
                Value out2Repl = b.create<tensor::ExtractOp>(
                    loc, shiftElemsLoop.getRegionIterArgs()[1],
                    ValueRange{idx0, nextIndVar});
                Value newVals = b.create<tensor::InsertOp>(
                    loc, shiftElemsLoop.getRegionIterArgs()[2],
                    shiftElemsLoop.getRegionIterArgs()[0],
                    ValueRange{idx0, nextIndVar});
                Value newInds = b.create<tensor::InsertOp>(
                    loc, shiftElemsLoop.getRegionIterArgs()[3],
                    shiftElemsLoop.getRegionIterArgs()[1],
                    ValueRange{idx0, nextIndVar});
                SmallVector<Value> newOperands;
                newOperands.push_back(newVals);
                newOperands.push_back(newInds);
                newOperands.push_back(out1Repl);
                newOperands.push_back(out2Repl);
                b.create<scf::YieldOp>(loc, newOperands);
              }

              lastLoopOp.erase();

              // Get the smellest element as the last added element.
              smElemElse = b.create<tensor::ExtractOp>(
                  loc, shiftElemsLoop.getResult(0),
                  ValueRange{idx0, ifExpand.getResult(0)});
              addElemsElse =
                  b.create<arith::AddIOp>(loc, ifExpand.getResult(0), one);
              outValsElse = shiftElemsLoop.getResult(0);
              outIndsElse = shiftElemsLoop.getResult(1);

              // Insert the element in the output.
              outValsElse = b.create<tensor::InsertOp>(
                  loc, elemToInsertIfNeeded, outValsElse,
                  ValueRange{idx0, insertionIndex});
              outIndsElse = b.create<tensor::InsertOp>(
                  loc,
                  b.create<arith::IndexCastOp>(loc, b.getIntegerType(32),
                                               elemIndex),
                  outIndsElse, ValueRange{idx0, insertionIndex});
              SmallVector<Value> newOperandsElse;
              newOperandsElse.push_back(outValsElse);
              newOperandsElse.push_back(outIndsElse);
              newOperandsElse.push_back(smElemElse);
              newOperandsElse.push_back(addElemsElse);
              b.create<scf::YieldOp>(loc, newOperandsElse);
            });

        Value outValsAtEnd = ifAddAtEnd.getResult(0);
        Value outIndsAtEnd = ifAddAtEnd.getResult(1);
        Value smElemAtEnd = ifAddAtEnd.getResult(2);
        Value addElemsAtEnd = ifAddAtEnd.getResult(3);

        SmallVector<Value> newOperands;
        newOperands.push_back(outValsAtEnd);
        newOperands.push_back(outIndsAtEnd);
        newOperands.push_back(smElemAtEnd);
        newOperands.push_back(addElemsAtEnd);
        b.create<scf::YieldOp>(loc, newOperands);
      });
  return InsertElemContinuationValues{
      ifAddElem.getResult(0), ifAddElem.getResult(1), ifAddElem.getResult(2),
      ifAddElem.getResult(3)};
}

/// Vectorize a `topKOp` with (1) static result and input types
static LogicalResult
vectorizeAsLinalgExtTopK(RewriterBase &rewriter, IREE::LinalgExt::TopkOp topkOp,
                         ArrayRef<int64_t> inputVectorSizes,
                         SmallVectorImpl<Value> &newResults) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(topkOp);
  Location loc = topkOp.getLoc();
  bool expectedRanksAndDims = true;
  ReifiedRankedShapedTypeDims reifiedReturnShapes;
  LogicalResult status =
      cast<ReifyRankedShapedTypeOpInterface>(topkOp.getOperation())
          .reifyResultShapes(rewriter, reifiedReturnShapes);
  assert(succeeded(status) && "failed to reify result shapes");

  if (failed(status))
    return failure();

  auto out0 = topkOp.getResults()[0];
  auto resShapedTy0 = llvm::cast<ShapedType>(out0.getType());
  auto out1 = topkOp.getResults()[1];
  auto resShapedTy1 = llvm::cast<ShapedType>(out1.getType());

  scf::ForOp scfInputLoop =
      dyn_cast<scf::ForOp>(topkOp->getParentRegion()->getParentOp());
  if (!scfInputLoop)
    return failure();

  auto outIdxElemType = resShapedTy1.getElementType();
  auto outValElemType = resShapedTy0.getElementType();

  {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(scfInputLoop);
    // Create a new loop based on the old one with adding the necessary
    // iter_args.
    if (failed(addIterArgs(rewriter, loc, scfInputLoop, outValElemType,
                           outIdxElemType)))
      return failure();
  }

  // Get the newly created loop.
  auto regParent = topkOp->getParentRegion()->getParentOp();
  if (!isa<scf::ForOp>(regParent))
    return failure();
  scf::ForOp newLoop = dyn_cast<scf::ForOp>(regParent);
  Value outputInitialized = newLoop.getRegionIterArgs()[2];
  // The following if is used to initialize the out arrays.
  // The outputs are the out0, out1, the smallest element added to the out0,
  // number of added elements.
  auto ifInitOut = rewriter.create<scf::IfOp>(
      loc, outputInitialized,
      [&](OpBuilder &b, Location loc) {
        Value outVals = newLoop.getRegionIterArgs()[0];
        // Out indexes
        Value outInds = newLoop.getRegionIterArgs()[1];
        Value idx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value firstInElem = rewriter.create<tensor::ExtractOp>(
            loc, topkOp.values(), ValueRange{idx0, idx0});
        auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
        // The operands are:
        // 1. OutValues
        // 2. OutIndeces
        // 3. smallest added element.
        // 4. Number of added elements
        SmallVector<Value> operands;
        operands.push_back(outVals);
        operands.push_back(outInds);
        operands.push_back(firstInElem);
        operands.push_back(zero);
        b.create<scf::YieldOp>(loc, operands);
      },
      [&](OpBuilder &b, Location loc) {
        // It is initialize. Just return the init_args values.
        Value outVals = newLoop.getRegionIterArgs()[0];
        // Out indexes
        Value outInds = newLoop.getRegionIterArgs()[1];
        SmallVector<Value> operands;
        // If the smallest element has been initialized,
        // Don't do anything.
        operands.push_back(outVals);
        operands.push_back(outInds);
        operands.push_back(newLoop.getRegionIterArgs()[3]);
        operands.push_back(newLoop.getRegionIterArgs()[4]);
        b.create<scf::YieldOp>(loc, operands);
      });

  Type inElemTy = topkOp.getInputType().getElementType();
  Value inValsVec = createReadOrMaskedRead(rewriter, loc, topkOp.getInputs()[0],
                                           inputVectorSizes, inElemTy);
  auto elemVecType = VectorType::get(inputVectorSizes, inElemTy);
  Value smallestMask = rewriter.create<vector::BroadcastOp>(
      loc, elemVecType, ifInitOut.getResult(2));
  Value comparedMask;
  if (auto floatType = llvm::dyn_cast_if_present<FloatType>(outValElemType)) {
    comparedMask = rewriter.create<arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OGT, inValsVec, smallestMask);
  } else if (auto intType =
                 llvm::dyn_cast_if_present<IntegerType>(outValElemType)) {
    comparedMask = rewriter.create<arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::sgt, inValsVec, smallestMask);
  } else {
    return failure();
  }

  Value outNumElems =
      rewriter.create<arith::ConstantIndexOp>(loc, resShapedTy1.getDimSize(1));
  // If the output is not filled yet, set the masks to true.
  Value cmpAddedElems =
      rewriter.create<arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
                                     ifInitOut.getResult(3), outNumElems);
  auto ifFillMask = rewriter.create<scf::IfOp>(
      loc, cmpAddedElems,
      [&](OpBuilder &b, Location loc) {
        Value tVal = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
        Value allElemsMask = rewriter.create<vector::BroadcastOp>(
            loc, comparedMask.getType(), tVal);
        SmallVector<Value> operands;
        operands.push_back(allElemsMask);
        b.create<scf::YieldOp>(loc, operands);
      },
      [&](OpBuilder &b, Location loc) {
        // else
        SmallVector<Value> operands;
        operands.push_back(comparedMask);
        b.create<scf::YieldOp>(loc, operands);
      });

  auto vecCmpCond = rewriter.create<vector::MultiDimReductionOp>(
      loc, ifFillMask.getResult(0),
      rewriter.create<arith::ConstantIntOp>(loc, 0, 1), SmallVector<bool>{1, 1},
      vector::CombiningKind::OR);

  auto ifVecCmp = rewriter.create<scf::IfOp>(
      loc, vecCmpCond,
      [&](OpBuilder &b, Location loc) {
        // There are bigger elemnets.
        SmallVector<Value> maskProcessingLoopOps;
        maskProcessingLoopOps.push_back(ifInitOut.getResult(0));
        maskProcessingLoopOps.push_back(ifInitOut.getResult(1));
        maskProcessingLoopOps.push_back(ifInitOut.getResult(2));
        maskProcessingLoopOps.push_back(ifInitOut.getResult(3));
        scf::YieldOp forYield;
        scf::ForOp maskProcessingLoop = b.create<scf::ForOp>(
            loc, b.create<arith::ConstantIndexOp>(loc, 0), newLoop.getStep(),
            b.create<arith::ConstantIndexOp>(loc, 1), maskProcessingLoopOps,
            [&](OpBuilder &bIn, Location nestedLoc, Value iv, ValueRange args) {
              forYield = bIn.create<scf::YieldOp>(loc);
            });

        {
          OpBuilder::InsertionGuard g(rewriter);
          rewriter.setInsertionPoint(forYield);
          SmallVector<OpFoldResult, 2> extractionIndices;
          extractionIndices.push_back(rewriter.getIndexAttr(0));
          extractionIndices.push_back(maskProcessingLoop.getInductionVar());
          auto sourceVectorType =
              dyn_cast<VectorType>(ifFillMask.getResult(0).getType());
          if (!sourceVectorType)
            expectedRanksAndDims = false;

          // The type here is always in the form 1x16xi1
          if (sourceVectorType.getRank() != 2)
            expectedRanksAndDims = false;

          bool hasLeadingDimUnitFixed =
              ((sourceVectorType.getShape().front() == 1) &&
               (!sourceVectorType.getScalableDims().front()));
          if (!hasLeadingDimUnitFixed)
            expectedRanksAndDims = false;
          VectorType newVType =
              VectorType::Builder(sourceVectorType).dropDim(0);

          // Drop leading/trailing unit dim by applying vector.shape_cast to all
          // operands
          Value aCast = rewriter.create<vector::ShapeCastOp>(
              loc, newVType, ifFillMask.getResult(0));
          auto elemMask = rewriter.create<vector::ExtractElementOp>(
              loc, aCast, maskProcessingLoop.getInductionVar());
          Value goToOut = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, elemMask,
              rewriter.create<arith::ConstantIntOp>(loc, 1, 1));
          auto insertElemIfNeeded = rewriter.create<scf::IfOp>(
              loc, goToOut,
              [&](OpBuilder &bIn2, Location loc) {
                inValsVec.print(llvm::errs());
                auto sourceVectorType =
                    dyn_cast<VectorType>(inValsVec.getType());
                if (!sourceVectorType)
                  expectedRanksAndDims = false;
                // The type here is always in the form 1x16xf/i32
                if (sourceVectorType.getRank() != 2)
                  expectedRanksAndDims = false;
                bool hasLeadingDimUnitFixed =
                    ((sourceVectorType.getShape().front() == 1) &&
                     (!sourceVectorType.getScalableDims().front()));
                if (!hasLeadingDimUnitFixed)
                  expectedRanksAndDims = false;
                VectorType newVType =
                    VectorType::Builder(sourceVectorType).dropDim(0);
                // Drop leading/trailing unit dim by applying vector.shape_cast
                // to all operands
                Value aCast =
                    bIn2.create<vector::ShapeCastOp>(loc, newVType, inValsVec);
                auto elemToInsertIfNeeded =
                    bIn2.create<vector::ExtractElementOp>(
                        loc, aCast, maskProcessingLoop.getInductionVar());
                Value elemIndex = bIn2.create<arith::AddIOp>(
                    loc, maskProcessingLoop.getInductionVar(),
                    newLoop.getInductionVar());
                InsertElemContinuationValues cont = insertElemInOutput(
                    loc, bIn2, elemToInsertIfNeeded, elemIndex, outNumElems,
                    InsertElemContinuationValues{
                        maskProcessingLoop.getRegionIterArgs()[0],
                        maskProcessingLoop.getRegionIterArgs()[1],
                        maskProcessingLoop.getRegionIterArgs()[2],
                        maskProcessingLoop.getRegionIterArgs()[3]});
                SmallVector<Value> operands;
                operands.push_back(cont.out0);
                operands.push_back(cont.out1);
                operands.push_back(cont.smallestElem);
                operands.push_back(cont.addedElems);
                bIn2.create<scf::YieldOp>(loc, operands);
              },
              [&](OpBuilder &bIn3, Location loc) {
                // else No need to insert anything.
                SmallVector<Value> operands;
                operands.push_back(maskProcessingLoop.getRegionIterArgs()[0]);
                operands.push_back(maskProcessingLoop.getRegionIterArgs()[1]);
                operands.push_back(maskProcessingLoop.getRegionIterArgs()[2]);
                operands.push_back(maskProcessingLoop.getRegionIterArgs()[3]);
                bIn3.create<scf::YieldOp>(loc, operands);
              });
          SmallVector<Value> newOperands;
          newOperands.push_back(insertElemIfNeeded.getResult(0));
          newOperands.push_back(insertElemIfNeeded.getResult(1));
          newOperands.push_back(insertElemIfNeeded.getResult(2));
          newOperands.push_back(insertElemIfNeeded.getResult(3));
          rewriter.create<scf::YieldOp>(loc, newOperands);
          forYield.erase();
        }
        SmallVector<Value> operands;
        operands.push_back(maskProcessingLoop.getResult(0));
        operands.push_back(maskProcessingLoop.getResult(1));
        operands.push_back(maskProcessingLoop.getResult(2));
        operands.push_back(maskProcessingLoop.getResult(3));
        b.create<scf::YieldOp>(loc, operands);
      },
      [&](OpBuilder &b, Location loc) {
        // else
        // No bigger elements. No work to do.
        SmallVector<Value> operands;
        operands.push_back(ifInitOut.getResult(0));
        operands.push_back(ifInitOut.getResult(1));
        operands.push_back(ifInitOut.getResult(2));
        operands.push_back(ifInitOut.getResult(3));
        b.create<scf::YieldOp>(loc, operands);
      });

  // If the types were not expected, don't do rewriting.
  if (!expectedRanksAndDims)
    return failure();

  // Create a new yield op for the loop.
  {
    OpBuilder::InsertionGuard g(rewriter);
    scf::YieldOp yieldOp =
        llvm::cast<scf::YieldOp>(newLoop.getBody()->getTerminator());
    rewriter.setInsertionPoint(yieldOp);
    SmallVector<Value> operands;
    operands.push_back(ifVecCmp.getResult(0));
    operands.push_back(ifVecCmp.getResult(1));
    // After first run the out arrays are initialized.
    operands.push_back(
        rewriter.create<arith::ConstantIntOp>(yieldOp.getLoc(), 0, 1));
    operands.push_back(ifVecCmp.getResult(2));
    operands.push_back(ifVecCmp.getResult(3));
    rewriter.create<scf::YieldOp>(yieldOp.getLoc(), operands);
    rewriter.eraseOp(yieldOp);
  }
  newResults.push_back(topkOp.outputValues());
  newResults.push_back(topkOp.outputIndices());
  return success();
}

void GenericVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();
  IRRewriter rewriter(context);
  SmallVector<Operation *> candidates;
  funcOp.walk([&](Operation *op) {
    if (isa<linalg::LinalgOp>(op)) {
      candidates.push_back(op);
    } else if (vectorizePadding && enableVectorMasking &&
               isa<tensor::PadOp>(op)) {
      candidates.push_back(op);
    } else if (enableVectorMasking &&
               isa<tensor::PackOp, tensor::UnPackOp>(op)) {
      candidates.push_back(op);
    } else if (isa<IREE::LinalgExt::TopkOp>(op)) {
      candidates.push_back(op);
    }
  });

  // The vector input sizes inference needs to use producers, so we apply
  // vectorization from bottom to top.
  std::reverse(candidates.begin(), candidates.end());
  for (Operation *op : candidates) {
    SmallVector<int64_t> vectorSizes;
    SmallVector<bool> scalableVecDims;
    if (enableVectorMasking) {
      std::optional<SizesAndScalableFlags> vectorSizesAndScalableDims =
          getVectorSizes(op, useConfiguredVectorSizes);
      if (vectorSizesAndScalableDims) {
        auto [sizes, scalableDims] = *vectorSizesAndScalableDims;
        vectorSizes.append(sizes.begin(), sizes.end());
        scalableVecDims.append(scalableDims.begin(), scalableDims.end());
      }
    }

    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      // Do not vectorize the op if the vector size is greater than or equal
      // to limit.
      if (enableVectorMasking) {
        if (std::accumulate(vectorSizes.begin(), vectorSizes.end(), 1,
                            std::multiplies<int64_t>()) >= maxVectorSize) {
          continue;
        }
      } else {
        if (failed(isWithinVectorSizeLimit(linalgOp, maxVectorSize))) {
          continue;
        }
      }
    } else if (auto topkOp = dyn_cast<IREE::LinalgExt::TopkOp>(op)) {
      auto arg0 = topkOp.getInputs()[0];
      auto shapedTy = llvm::cast<ShapedType>(arg0.getType());
      vectorSizes.append(shapedTy.getShape().begin(),
                         shapedTy.getShape().end());
      if (shapedTy.isDynamicShape(vectorSizes))
        continue;
      if (failed(vectorizeTopkOpPrecondition(topkOp, vectorSizes))) {
        VEC_LDBG("Vectorization TopK pre-conditions failed\n");
        return; // falied.
      }
      SmallVector<Value> results;
      if (failed(vectorizeAsLinalgExtTopK(rewriter, topkOp, vectorSizes,
                                          results))) {
        VEC_LDBG("TopK Vectorization failed\n");
        return;
      }
      if (!results.empty())
        rewriter.replaceOp(op, results);
      else
        rewriter.eraseOp(op);

      return;
    }

    // Pad scalable dims with `false` to match the vector sizes.
    scalableVecDims.resize(vectorSizes.size());
    (void)linalg::vectorize(rewriter, op, vectorSizes, scalableVecDims,
                            vectorizeGatherAccesses);
  };

  {
    // Canonicalize mask related ops before we lower them.
    RewritePatternSet maskCanonPatterns(funcOp.getContext());
    vector::CreateMaskOp::getCanonicalizationPatterns(maskCanonPatterns,
                                                      funcOp.getContext());
    vector::ConstantMaskOp::getCanonicalizationPatterns(maskCanonPatterns,
                                                        funcOp.getContext());
    vector::MaskOp::getCanonicalizationPatterns(maskCanonPatterns,
                                                funcOp.getContext());
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(maskCanonPatterns)))) {
      return signalPassFailure();
    }
  }

  // TODO: Move this down the pipeline once we have the ODM-based masking
  // representation.
  RewritePatternSet vectorizationPatterns(funcOp.getContext());
  if (generateContract) {
    vector::populateVectorTransferPermutationMapLoweringPatterns(
        vectorizationPatterns);
    vector::populateVectorReductionToContractPatterns(vectorizationPatterns);
  }
  if (foldCastIntoContract) {
    vector::populateFoldArithExtensionPatterns(vectorizationPatterns);
  }
  if (enableVectorMasking) {
    vector::populateVectorMaskLoweringPatternsForSideEffectingOps(
        vectorizationPatterns);
    vectorizationPatterns.add<linalg::LinalgCopyVTRForwardingPattern,
                              linalg::LinalgCopyVTWForwardingPattern>(
        funcOp.getContext(), /*benefit=*/2);
  }

  if (enableCleanup) {
    vector::TransferReadOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                        funcOp.getContext());
    vector::TransferWriteOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                         funcOp.getContext());
  }
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(vectorizationPatterns));

  // Apply the pad tensor op vectorization separately to avoid running the
  // GenericPadOpVectorizationPattern too early.
  // TODO: Improve once we have better infrastructure to control pattern
  // application.
  if (vectorizePadding) {
    RewritePatternSet patterns(funcOp.getContext());
    linalg::populatePadOpVectorizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
}

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGenericVectorizationPass() {
  return std::make_unique<GenericVectorizationPass>();
}
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGenericVectorizationPass(const GenericVectorizationPassOptions &options) {
  return std::make_unique<GenericVectorizationPass>(options);
}

} // namespace mlir::iree_compiler
