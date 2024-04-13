// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- PropagateDataLayout.cpp - Pass to propagate data layout to globals -===//
//
// The pass is to propagate data layout operations like tensor.pack all the
// way to global loads/stores, and update the layout of globals
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/GlobalOptimization/DataLayoutUtils.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-global-opt-propagate-data-layout"

namespace mlir::iree_compiler::GlobalOptimization {

template <typename T>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const llvm::SmallVectorImpl<T> &vector) {
  os << "[ ";
  for (T element : vector) {
    os << element << " ";
  }
  os << "]";

  return os;
}

template <typename T>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const llvm::ArrayRef<T> &vector) {
  os << "[ ";
  for (T element : vector) {
    os << element << " ";
  }
  os << "]";

  return os;
}

static llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, const DataLayoutTransformation &transform) {
  os << "originalType: " << transform.getOriginalType() << "\n";
  os << "transformedType: " << transform.getTransformedType() << "\n";
  os << "innerDimsPos: " << transform.getInnerDimsPos() << "\n";
  os << "innerTileSizes: " << transform.getInnerTileSizes() << "\n";
  os << "outerDimsPerm: " << transform.getOuterDimsPerm() << "\n";
  os << "constantPadValue: " << transform.getConstantPadValue() << "\n";
  os << "correspondingTransformedIndices: "
     << transform.getCorrespondingTransformedIndices() << "\n";
  return os;
}

class GlobalDataLayoutState : public DFX::AbstractState {
public:
  ~GlobalDataLayoutState() {
    for (auto id : getLayoutIDs()) {
      delete layoutMap[id];
    }
  }

  bool isValidState() const override { return true; }
  /// TODO: There are cases where the `correspondingTransformedIndices` of a
  /// transformation may not have maximal information when transforms are
  /// propagated from the first neighbor that has a valid transform. To fix
  /// this, the Fixpoint state should be based on how many unknown indices there
  /// are in `correspondingTransformedIndices`, and states need to be allowed to
  /// propagate `correspondingTransformedIndices` information after they already
  /// have a valid transform.
  bool isAtFixpoint() const override {
    return false;
    return getLayoutIDs().size() && all_of(getLayoutIDs(), [&](StringRef id) {
             return layoutMap.lookup(id)->hasValidTransform();
           });
  }

  ChangeStatus indicateOptimisticFixpoint() override {
    return ChangeStatus::UNCHANGED;
  }

  ChangeStatus indicatePessimisticFixpoint() override {
    return ChangeStatus::CHANGED;
  }

  SmallVector<StringRef> getLayoutIDs() const {
    SetVector<StringRef> ids;
    for (auto it : layoutMap) {
      ids.insert(it.first);
    }
    return ids.takeVector();
  };
  bool hasLayoutID(StringRef id) { return layoutMap.contains(id); };
  DataLayoutTransformation *getDataLayoutTransformation(StringRef id) {
    if (layoutMap.contains(id))
      return layoutMap[id];
    return nullptr;
  };
  DataLayoutNodeType getNodeType() const { return nodeType; };
  bool addDataLayoutTransformation(StringRef id,
                                   DataLayoutTransformation *newLayout) {
    return layoutMap.insert(std::make_pair(id, newLayout)).second;
  };
  void setDataLayoutTransformation(StringRef id,
                                   DataLayoutTransformation *newLayout) {
    if (layoutMap.count(id)) {
      delete layoutMap[id];
      layoutMap.erase(id);
    }
    layoutMap.insert(std::make_pair(id, newLayout));
  };

  bool initializeTerminalNodeIDs(Value value) {
    SmallVector<StringRef> IDs = getTerminalNodeIDs(value);
    bool addedID = false;
    for (auto id : IDs) {
      DataLayoutTransformation *newLayout =
          new DataLayoutTransformation(cast<ShapedType>(value.getType()));
      if (addDataLayoutTransformation(id, newLayout))
        addedID = true;
      else
        delete newLayout;
    }
    return addedID;
  }

  bool initializeTerminalNodeLayouts(Value value) {
    bool changed = false;
    auto layoutType = cast<ShapedType>(value.getType());
    SmallVector<StringRef> IDs = getTerminalNodeIDs(value);
    for (auto id : IDs) {
      if (getDataLayoutTransformation(id) &&
          !getDataLayoutTransformation(id)->hasValidTransform()) {
        changed = true;
      }
      DataLayoutTransformation *newLayout =
          DataLayoutTransformation::getIdentityTransformation(layoutType);
      setDataLayoutTransformation(id, newLayout);
    }
    return changed;
  }

  void setNodeType(DataLayoutNodeType newNodeType) { nodeType = newNodeType; };

private:
  DenseMap<StringRef, DataLayoutTransformation *> layoutMap;
  DataLayoutNodeType nodeType = DataLayoutNodeType::UNINITIALIZED;
};

class GlobalDataLayoutValueElement
    : public DFX::StateWrapper<GlobalDataLayoutState, DFX::ValueElement> {
public:
  using BaseType = DFX::StateWrapper<GlobalDataLayoutState, DFX::ValueElement>;
  using BaseType::BaseType;

  static GlobalDataLayoutValueElement &createForPosition(const Position &pos,
                                                         DFX::Solver &solver) {
    return *(new (solver.getAllocator()) GlobalDataLayoutValueElement(pos));
  }

  // Identity definitions.
  static const char ID;
  const std::string getName() const override {
    return "GlobalDataLayoutValueElement";
  }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  const std::string getAsStr(AsmState &asmState) const override;

  /// TODO: Restrict BFS search to not search through barriers.
  static SetVector<Value> getTensorValueNeighbors(Value value) {
    SetVector<Value> connectedValues;
    auto appendOpValues = [&](Operation *op) {
      for (Value v : op->getOperands()) {
        if (isa<ShapedType>(v.getType())) {
          connectedValues.insert(v);
        }
      }
      for (Value v : op->getResults()) {
        if (isa<ShapedType>(v.getType())) {
          connectedValues.insert(v);
        }
      }
    };
    // Get defining op values
    if (Operation *definingOp = value.getDefiningOp()) {
      appendOpValues(definingOp);
    }
    // Get user op values
    for (Operation *user : value.getUsers()) {
      appendOpValues(user);
    }
    connectedValues.remove(value);
    return connectedValues;
  }

  void walkAllConnectedTensorValues(std::function<void(Value &)> fn) {
    DenseSet<Value> visitedValues;
    recursivelyWalkAllConnectedTensorValues(getValue(), fn, visitedValues);
  }

private:
  static void
  recursivelyWalkAllConnectedTensorValues(Value value,
                                          std::function<void(Value &)> fn,
                                          DenseSet<Value> &visitedValues) {
    if (visitedValues.contains(value)) {
      return;
    }
    visitedValues.insert(value);
    fn(value);
    if (getNodeTypeForValue(value) == DataLayoutNodeType::BARRIER) {
      return;
    }
    for (Value v : getTensorValueNeighbors(value)) {
      recursivelyWalkAllConnectedTensorValues(v, fn, visitedValues);
    }
  }
  void initializeValue(Value value, DFX::Solver &solver) override;
  ChangeStatus updateValue(Value value, DFX::Solver &solver) override;
};
const char GlobalDataLayoutValueElement::ID = 0;

const std::string
GlobalDataLayoutValueElement::getAsStr(AsmState &asmState) const {
  std::string s("getAsStr unimplemented");
  return s;
}

class GlobalDataLayoutAnalysis {
public:
  GlobalDataLayoutAnalysis(ModuleOp rootOp)
      : explorer(rootOp, TraversalAction::SHALLOW),
        solver(explorer, allocator) {
    explorer.setOpAction<IREE::Util::FuncOp>(TraversalAction::RECURSE);
    explorer.setOpAction<IREE::Util::InitializerOp>(TraversalAction::RECURSE);
    explorer.initialize();
  }

  SmallVector<Value> getGlobalLayoutEndpoints() {
    SmallVector<Value> endpoints;
    explorer.forEachGlobal([&](const Explorer::GlobalInfo *globalInfo) {
      auto global = globalInfo->op;
      auto tensorType = dyn_cast<RankedTensorType>(global.getGlobalType());
      if (!tensorType || !global.isGlobalMutable() || !global.isGlobalPrivate())
        return;
      for (auto load : globalInfo->getLoads())
        endpoints.push_back(load.getLoadedGlobalValue());
      for (auto store : globalInfo->getStores())
        endpoints.push_back(store.getStoredGlobalValue());
    });
    return endpoints;
  }

  LogicalResult run() {
    SmallVector<Value> endpoints = getGlobalLayoutEndpoints();
    for (auto endpoint : endpoints) {
      solver.getOrCreateElementFor<GlobalDataLayoutValueElement>(
          Position::forValue(endpoint));
    }
    return solver.run();
  }

  /// Returns a list of all tensor values that may incur an element-wise cost
  /// when transformed with the given transformation.
  SetVector<Value>
  getTransformedValues(DataLayoutTransformation *tf,
                       SmallVector<GlobalDataLayoutValueElement *> costNodes,
                       StringRef layoutID) {
    SetVector<Value> transformedVals;
    for (auto node : costNodes) {
      auto definingOp = node->getValue().getDefiningOp();
      if (llvm::isa_and_nonnull<tensor::EmptyOp>(definingOp)) {
        continue;
      }
      DataLayoutTransformation *costNodeTf =
          node->getState().getDataLayoutTransformation(layoutID);
      // If the transformation does not intersect the value's dimensions, then
      // the value does not actually need to be transformed.
      if (tf->isIntersecting(*costNodeTf)) {
        transformedVals.insert(node->getValue());
      }
    }
    return transformedVals;
  }

  /// Return true if `a` can be known to have a greater or equal number of
  /// elements than `b`, assuming dynamic dims must be at least equal to 1.
  bool hasMoreTotalElements(SetVector<Value> a, SetVector<Value> b) {
    int64_t staticCountA = 0, staticCountB = 0;
    bool hasDynamicA = false, hasDynamicB = false;
    auto countElements = [](SetVector<Value> set, SetVector<Value> other,
                            int64_t &staticCount, bool &hasDynamicSizes) {
      for (Value v : set) {
        // Ignore values that are in both sets
        if (!other.count(v)) {
          auto shape = cast<ShapedType>(v.getType()).getShape();
          int64_t count = 1;
          for (auto size : shape) {
            if (size == ShapedType::kDynamic) {
              hasDynamicSizes = true;
              continue;
            }
            count *= size;
          }
          // Always add to staticCount, even if there were dynamic sizes.
          // Dynamic sizes must be at least 1, so we can know statically
          // that there are at least [product of static dims] elements.
          staticCount += count;
        }
      }
    };
    countElements(a, b, staticCountA, hasDynamicA);
    countElements(b, a, staticCountB, hasDynamicB);
    // If B is not dynamic, and the static count of A is greater than B,
    // then we can know that A has at least as many total elements as B.
    if (!hasDynamicB && staticCountA >= staticCountB) {
      return true;
    }
    // Otherwise, B has a greater static count, or B is dynamic and we
    // cannot know statically if there are more elements in B.
    return false;
  }

  LogicalResult getSubgraphEdgeNodesAndBestLayoutTransformation(
      SetVector<GlobalDataLayoutValueElement *> subgraph,
      SmallVector<Value> &edgeNodes, DataLayoutTransformation *&transform,
      StringRef layoutID) {
    assert(edgeNodes.empty() && "edgeNodes expected to be empty");
    SmallVector<DataLayoutTransformation *> barrierTransforms;
    SmallVector<GlobalDataLayoutValueElement *> costNodes;
    DataLayoutTransformation *bestTf;
    for (auto node : subgraph) {
      GlobalDataLayoutState &state = node->getState();
      if (state.getNodeType() == DataLayoutNodeType::BARRIER) {
        costNodes.push_back(node);
        barrierTransforms.push_back(
            state.getDataLayoutTransformation(layoutID));
      }
      if (!getTerminalNodeIDs(node->getValue()).empty()) {
        edgeNodes.push_back(node->getValue());
        // Initialize the "bestTf" as the identity transformation at edge nodes.
        bestTf = state.getDataLayoutTransformation(layoutID);
      }
    }
    if (barrierTransforms.empty() || edgeNodes.empty()) {
      return failure();
    }

    SetVector<Value> bestTransformedValues =
        getTransformedValues(bestTf, costNodes, layoutID);
    for (auto tf : barrierTransforms) {
      SetVector<Value> transformedValues =
          getTransformedValues(tf, costNodes, layoutID);
      // Assume the cost of layout transformations are simple element-wise
      // operations, so the cost of the transformations are proportional to the
      // combined total number of elements in the set of transformed tensors.
      if (hasMoreTotalElements(transformedValues, bestTransformedValues)) {
        bestTf = tf;
        bestTransformedValues = transformedValues;
      }
    }
    LLVM_DEBUG({
      llvm::dbgs() << "layoutID: " << layoutID << "\n";
      llvm::dbgs() << "best tf: " << *bestTf << "\n\n";
    });
    if (bestTf->isIdentity())
      return failure();
    transform = bestTf;
    return success();
  }

  LogicalResult getEdgeNodesAndBestLayoutTransformations(
      SmallVector<SmallVector<Value>> &edgeNodes,
      SmallVector<DataLayoutTransformation *> &transforms,
      SmallVector<StringRef> &layoutIDs) {
    DenseMap<StringRef, SetVector<GlobalDataLayoutValueElement *>> subgraphMap;
    auto walkFn = [&](Value val) -> WalkResult {
      if (auto *elementPtr =
              solver.lookupElementFor<GlobalDataLayoutValueElement>(
                  Position::forValue(val),
                  /*queryingElement=*/nullptr, DFX::Resolution::NONE,
                  /*allowInvalidState=*/false)) {
        GlobalDataLayoutState &state = elementPtr->getState();
        // Only support subgraphs with a single layout ID for now
        auto stateLayoutIDs = state.getLayoutIDs();
        if (stateLayoutIDs.size() != 1)
          return WalkResult::advance();
        if (subgraphMap.count(stateLayoutIDs[0])) {
          subgraphMap[stateLayoutIDs[0]].insert(elementPtr);
        } else {
          SetVector<GlobalDataLayoutValueElement *> s;
          s.insert(elementPtr);
          subgraphMap.insert(std::make_pair(stateLayoutIDs[0], s));
        }
      }
      return WalkResult::advance();
    };
    if (explorer.walkValues(walkFn) == TraversalResult::INCOMPLETE) {
      return failure();
    }

    // For each subgraph, compute the optimal layout and endpoint nodes
    for (auto [layoutID, valueElems] : subgraphMap) {
      SmallVector<Value> subgraphEdgeNodes;
      DataLayoutTransformation *subgraphTransform;
      if (!failed(getSubgraphEdgeNodesAndBestLayoutTransformation(
              valueElems, subgraphEdgeNodes, subgraphTransform, layoutID))) {
        edgeNodes.push_back(subgraphEdgeNodes);
        transforms.push_back(subgraphTransform);
        layoutIDs.push_back(layoutID);
      }
    }
    if (edgeNodes.empty()) {
      return failure();
    }
    return success();
  }

  void annotateDataLayoutNodes() {
    auto walkFn = [&](Value val) -> WalkResult {
      if (auto definingOp = val.getDefiningOp()) {
        if (auto *elementPtr =
                solver.lookupElementFor<GlobalDataLayoutValueElement>(
                    Position::forValue(val),
                    /*queryingElement=*/nullptr, DFX::Resolution::NONE,
                    /*allowInvalidState=*/false)) {
          GlobalDataLayoutState &state = elementPtr->getState();
          setNodeTypeAttribute(definingOp, state.getNodeType());
          for (StringRef id : state.getLayoutIDs()) {
            setDataLayoutTransformationAttributes(
                definingOp, state.getDataLayoutTransformation(id), id);
          }
        }
      }
      return WalkResult::advance();
    };
    explorer.walkValues(walkFn);
  }

  const Explorer::GlobalInfo *getGlobalInfo(StringRef layoutID) {
    return explorer.queryGlobalInfoFrom(layoutID, explorer.getRootOp());
  }

private:
  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;
};

void GlobalDataLayoutValueElement::initializeValue(Value value,
                                                   DFX::Solver &solver) {
  GlobalDataLayoutState &newState = getState();
  if (newState.getNodeType() == DataLayoutNodeType::UNINITIALIZED) {
    DataLayoutNodeType newNodeType = getNodeTypeForValue(value);
    newState.setNodeType(newNodeType);
  }
  return;
}

ChangeStatus GlobalDataLayoutValueElement::updateValue(Value value,
                                                       DFX::Solver &solver) {
  GlobalDataLayoutState &newState = getState();
  ChangeStatus status = ChangeStatus::UNCHANGED;

  // Compute the DataLayoutTransformation transformation to the current value.
  SetVector<Value> valueNeighbors = getTensorValueNeighbors(value);
  for (Value neighbor : valueNeighbors) {
    // Find neighbors that are part of the layout graph
    auto *neighborVE = solver.lookupElementFor<GlobalDataLayoutValueElement>(
        Position::forValue(neighbor), /*queryingElement=*/nullptr,
        DFX::Resolution::NONE, /*allowInvalidState=*/false);
    if (!neighborVE) {
      continue;
    }
    GlobalDataLayoutState &neighborState = neighborVE->getState();
    for (StringRef id : neighborState.getLayoutIDs()) {
      auto *neighborLayout = neighborState.getDataLayoutTransformation(id);
      // Start by initializing the current newState with an empty transformation
      // if it does not already have one for this ID.
      auto *emptyTfLayout =
          new DataLayoutTransformation(neighborLayout->getOriginalType());
      if (newState.addDataLayoutTransformation(id, emptyTfLayout)) {
        status = ChangeStatus::CHANGED;
      } else {
        delete emptyTfLayout;
      }
      // Try to infer the transformation to the current value using the
      // transformation from the neighboring value.
      if (!neighborLayout->hasValidTransform()) {
        continue;
      }
      auto *newLayout = new DataLayoutTransformation(*neighborLayout);
      if (newLayout->transformLayout(neighbor, value)) {
        // If there is already a known transformation to the current node, then
        // try to combine the information from the new inferred layout with the
        // known transformation, since not all transformations will carry the
        // maximal information for a given value.
        auto currentTf = newState.getDataLayoutTransformation(id);
        if (currentTf && currentTf->hasValidTransform()) {
          if (currentTf->combineLayout(*newLayout)) {
            status = ChangeStatus::CHANGED;
          }
          delete newLayout;
          continue;
        }
        // Otherwise, take the inferred transformation.
        newState.setDataLayoutTransformation(id, newLayout);
        status = ChangeStatus::CHANGED;
      } else {
        delete newLayout;
      }
    }
  }

  // If the newState did not pick up any layoutIDs from its neighbors, then
  // this is the first node to be initialized with layoutIDs, and we need to
  // walk the graph to find all reachable layouts from this node. These layouts
  // will propagate through the neighboring nodes when the neighbors do a state
  // update, so this should only happen once per connected graph.
  if (newState.getLayoutIDs().size() == 0) {
    newState.initializeTerminalNodeIDs(value);
    walkAllConnectedTensorValues(
        [&newState](Value v) { newState.initializeTerminalNodeIDs(v); });
    if (newState.getLayoutIDs().size() > 0) {
      status = ChangeStatus::CHANGED;
    }
  }
  // Terminal nodes (sources of the layouts) initialize their corresponding
  // layoutID with the identity transformation.
  if (newState.initializeTerminalNodeLayouts(value)) {
    status = ChangeStatus::CHANGED;
  }

  // Intermediate nodes add ValueElements for all neighboring tensor values.
  if (newState.getNodeType() == DataLayoutNodeType::INTERMEDIATE) {
    for (Value val : valueNeighbors) {
      if (val != value) {
        if (val.getDefiningOp<tensor::EmptyOp>()) {
          continue;
        }
        auto &newNode = solver.getElementFor<GlobalDataLayoutValueElement>(
            *this, Position::forValue(val), DFX::Resolution::REQUIRED);
        solver.recordDependence(*this, newNode, DFX::Resolution::REQUIRED);
      }
    }
  }

  return status;
}

//===----------------------------------------------------------------------===//
// Propagation pattern definitions
//===----------------------------------------------------------------------===//

class FoldCancellingUnPackPackOps final
    : public OpRewritePattern<tensor::UnPackOp> {
public:
  using OpRewritePattern<tensor::UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::UnPackOp unpackOp,
                                PatternRewriter &rewriter) const override {
    return tensor::UnPackOp::canonicalize(unpackOp, rewriter);
  }
};

static bool haveSameTiles(tensor::PackOp packOp, tensor::UnPackOp unPackOp) {
  auto packTiles = packOp.getMixedTiles();
  auto unPackTiles = unPackOp.getMixedTiles();
  if (packTiles.size() != unPackTiles.size())
    return false;
  for (size_t i = 0, e = packTiles.size(); i < e; i++) {
    if (!isEqualConstantIntOrValue(packTiles[i], unPackTiles[i]))
      return false;
  }
  return true;
}

class FoldCancellingPackUnPackOps final
    : public OpRewritePattern<tensor::PackOp> {
public:
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    // Fold an unpack(pack(x)) to x, ignoring padding_value if explicitly.
    // labeled as a foldable unpack.
    if (auto unPackOp = packOp.getSource().getDefiningOp<tensor::UnPackOp>()) {
      if (!hasFoldablePackUnPackAttribute(unPackOp)) {
        return failure();
      }
      if (unPackOp.getSourceType() != packOp.getDestType())
        return failure();
      if (packOp.getInnerDimsPos() != unPackOp.getInnerDimsPos() ||
          packOp.getOuterDimsPerm() != unPackOp.getOuterDimsPerm() ||
          !haveSameTiles(packOp, unPackOp))
        return failure();
      rewriter.replaceOp(packOp, unPackOp.getSource());
      return success();
    }
    return failure();
  }
};

class BubbleUpTensorExtractSliceThroughUnPack final
    : public OpRewritePattern<tensor::ExtractSliceOp> {
public:
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter &rewriter) const override {
    auto unpackOp =
        extractSliceOp.getSource().getDefiningOp<tensor::UnPackOp>();
    if (!unpackOp) {
      return failure();
    }
    auto nodeType = getNodeTypeFromAttr(extractSliceOp);
    if (!nodeType || nodeType.value() == DataLayoutNodeType::BARRIER) {
      return failure();
    }

    Location loc = extractSliceOp->getLoc();
    auto mixedTiles = unpackOp.getMixedTiles();
    SmallVector<int64_t> innerDimsPos(unpackOp.getInnerDimsPos());
    SmallVector<int64_t> outerDimsPerm(unpackOp.getOuterDimsPerm());

    std::optional<llvm::SmallDenseSet<unsigned>> maybeRankReducingMask =
        mlir::computeRankReductionMask(
            extractSliceOp.getStaticSizes(),
            extractSliceOp.getResultType().getShape());
    if (!maybeRankReducingMask) {
      return rewriter.notifyMatchFailure(
          extractSliceOp,
          "failed to compute rank reducing mask for packed extract_slice");
    }
    auto rankReducingMask = maybeRankReducingMask.value();

    SmallVector<OpFoldResult> newMixedOffsets(extractSliceOp.getMixedOffsets());
    SmallVector<OpFoldResult> newMixedSizes(extractSliceOp.getMixedSizes());
    SmallVector<OpFoldResult> newMixedStrides(extractSliceOp.getMixedStrides());

    if (failed(getPackedSliceMetadata(rewriter, loc, mixedTiles, innerDimsPos,
                                      outerDimsPerm, rankReducingMask,
                                      newMixedOffsets, newMixedSizes,
                                      newMixedStrides))) {
      return failure();
    }

    // Create the packed ExtractSliceOp.
    RankedTensorType packedExtractResultType =
        getPackedSliceType(unpackOp.getSourceType(), newMixedSizes,
                           rankReducingMask, outerDimsPerm, innerDimsPos);
    Value packedExtractOp = rewriter.create<tensor::ExtractSliceOp>(
        loc, packedExtractResultType, unpackOp.getSource(), newMixedOffsets,
        newMixedSizes, newMixedStrides);

    // Unpack the extracted slice.
    // This tensor can have unit dims in some packed dimensions. These need to
    // be collapsed before unpacking.
    SmallVector<OpFoldResult> sliceInnerTiles(
        newMixedSizes.end() - mixedTiles.size(), newMixedSizes.end());
    FailureOr<Value> unpackedSlice =
        unPackSliceOfTensor(rewriter, packedExtractOp, sliceInnerTiles,
                            maybeRankReducingMask.value(), unpackOp,
                            extractSliceOp.getMixedSizes());
    if (failed(unpackedSlice)) {
      return rewriter.notifyMatchFailure(extractSliceOp,
                                         "failed to unpack extracted slice");
    }
    rewriter.replaceOp(extractSliceOp, unpackedSlice.value());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace {

struct PropagateDataLayoutPass
    : public PropagateDataLayoutBase<PropagateDataLayoutPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    IREE::Flow::FlowDialect>();
  }

  void runOnOperation() override;
};

} // namespace

void PropagateDataLayoutPass::runOnOperation() {
  auto moduleOp = getOperation();

  // Do data layout analysis and rewrite globals
  {

    GlobalDataLayoutAnalysis analysis(moduleOp);
    if (failed(analysis.run())) {
      LLVM_DEBUG({ llvm::dbgs() << "analysis failed\n"; });
      return signalPassFailure();
    }

    analysis.annotateDataLayoutNodes();

    LLVM_DEBUG({
      llvm::dbgs() << "\n--- After annotating DAG with start/end points ---\n";
      moduleOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Compute optimal layout for each global
    SmallVector<SmallVector<Value>> edgeNodes;
    SmallVector<DataLayoutTransformation *> transforms;
    SmallVector<StringRef> layoutIDs;
    if (failed(analysis.getEdgeNodesAndBestLayoutTransformations(
            edgeNodes, transforms, layoutIDs))) {
      LLVM_DEBUG({ llvm::dbgs() << "Could not compute optimal layouts\n"; });
      return;
    }

    {
      SymbolTable moduleSymbols(moduleOp);
      IRRewriter rewriter(&getContext());
      for (auto [idx, layoutID] : llvm::enumerate(layoutIDs)) {
        if (failed(transformGlobalsToNewLayout(
                rewriter, edgeNodes[idx], transforms[idx],
                analysis.getGlobalInfo(layoutID), moduleSymbols))) {
          return signalPassFailure();
        }
      }
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After rewriting globals into new layout ---\n";
    moduleOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Propagate data layout transformation to start/end nodes
  {
    MLIRContext *context = &getContext();
    RewritePatternSet propagationPatterns(context);
    propagationPatterns.insert<BubbleUpTensorExtractSliceThroughUnPack>(
        context);
    propagationPatterns.insert<FoldCancellingPackUnPackOps>(context);
    propagationPatterns.insert<FoldCancellingUnPackPackOps>(context);

    if (failed(applyPatternsAndFoldGreedily(moduleOp,
                                            std::move(propagationPatterns)))) {
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After propagating data layout through DAGs ---\n";
    moduleOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
}

std::unique_ptr<OperationPass<mlir::ModuleOp>> createPropagateDataLayoutPass() {
  return std::make_unique<PropagateDataLayoutPass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
