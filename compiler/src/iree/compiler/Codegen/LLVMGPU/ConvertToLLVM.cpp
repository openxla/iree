// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/ConvertToLLVM.h"

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmgpu-convert-to-llvm"
namespace mlir {
namespace iree_compiler {

static const StringLiteral kAllocAliasGroup = "__alloc_alias_group__";

LogicalResult verifyLLVMConversionCompatibility(ModuleOp moduleOp) {
  LogicalResult compatible = success();

  moduleOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
    auto memrefType = subspanOp.getType().dyn_cast<MemRefType>();
    if (memrefType) {
      Type elType = memrefType.getElementType();
      if (!elType.isa<FloatType, IntegerType>()) {
        subspanOp.emitError()
            << "only integer and floating point element types "
               "are supported at interface boundary";
        compatible = failure();
      }
    }
  });

  return compatible;
}

/// Sink allocs in the CFG to reduce its liverange.
void sinkAlloc(ModuleOp moduleOp) {
  DominanceInfo dominators(moduleOp);
  SmallVector<Operation *> sinkOps;
  moduleOp.walk([&](memref::AllocOp alloc) { sinkOps.push_back(alloc); });
  for (Operation *sinkOp : sinkOps) {
    Block *dom = nullptr;
    for (Operation *user : sinkOp->getUsers()) {
      if (!dom) {
        dom = user->getBlock();
        continue;
      }
      dom = dominators.findNearestCommonDominator(dom, user->getBlock());
    }
    llvm::SmallDenseSet<Operation *> users(sinkOp->getUsers().begin(),
                                           sinkOp->getUsers().end());
    Operation *firstUse = dom->getTerminator();
    for (Operation &op : dom->getOperations()) {
      if (users.count(&op)) {
        firstUse = &op;
        break;
      }
    }
    sinkOp->moveBefore(firstUse);
  }
}

/// Analyze the liverange of allocs and set them in individual groups if
/// possible and mark allocs with an attributes.
/// The algorithm is a simplistic memory allocation solution. It sorts
/// allocations into alias groups. Everytime two alloc's liverange interfers
/// they are merge into the same group. If a new alloc is part of multiple alias
/// groups all those are merged into one. At the end we are left with groups of
/// allocations that are disjoint and can use the same memory.
// TODO: Move this to a common place if needed by other backends.
void analyzeSharedMemoryAlloc(ModuleOp moduleOp) {
  // First sink the alloc as low as possible in the CFG.
  sinkAlloc(moduleOp);
  struct AllocGroup {
    std::vector<Operation *> allocs;
    // Keep track of every operation where any of the alloc in the group is
    // live.
    llvm::DenseSet<Operation *> liveness;
  };
  Liveness liveness(moduleOp);
  SmallVector<AllocGroup> groups;
  moduleOp.walk([&](memref::AllocOp alloc) {
    SmallVector<size_t> aliasGroups;
    for (size_t i : llvm::seq<size_t>(0, groups.size())) {
      AllocGroup &group = groups[i];
      for (Operation *user : alloc->getUsers()) {
        if (group.liveness.count(user)) {
          aliasGroups.push_back(i);
          break;
        }
      }
    }
    if (aliasGroups.empty()) {
      AllocGroup &newGroup = groups.emplace_back();
      newGroup.allocs.push_back(alloc);
      Liveness::OperationListT liveInfo = liveness.resolveLiveness(alloc);
      newGroup.liveness.insert(liveInfo.begin(), liveInfo.end());
    } else {
      // Merge the alloc into the first alias group it interfers with.
      AllocGroup &mergeGroup = groups[aliasGroups[0]];
      mergeGroup.allocs.push_back(alloc);
      Liveness::OperationListT liveInfo = liveness.resolveLiveness(alloc);
      mergeGroup.liveness.insert(liveInfo.begin(), liveInfo.end());
      // Then merge all the other alias groups into the first group.
      for (size_t i = 1, e = aliasGroups.size(); i < e; i++) {
        AllocGroup &group = groups[aliasGroups[i]];
        mergeGroup.allocs.insert(mergeGroup.allocs.end(), group.allocs.begin(),
                                 group.allocs.end());
        mergeGroup.liveness.insert(group.liveness.begin(),
                                   group.liveness.end());
        // For simplicity we leave the group empty and don't remove it.
        group.allocs.clear();
        group.liveness.clear();
      }
    }
  });

  LLVM_DEBUG({
    for (size_t i = 0; i < groups.size(); i++) {
      llvm::dbgs() << "Alias group " << i << ":\n";
      for (Operation *op : groups[i].allocs) op->dump();
    }
  });

  // Builder only used as attribute helper.
  OpBuilder builder(moduleOp.getContext());
  for (size_t i = 0; i < groups.size(); i++) {
    for (Operation *alloc : groups[i].allocs) {
      alloc->setAttr(kAllocAliasGroup, builder.getIndexAttr(i));
    }
  }
}

void ConvertToDynamicSharedMemory(ModuleOp moduleOp) {
  SymbolTableCollection symbolTableCollection;
  // Collect all the adressOfOps to static shared memory globals.
  SmallVector<LLVM::AddressOfOp> addressOfOps;
  moduleOp.walk([&](LLVM::AddressOfOp addressOfOp) {
    // Check that the global associated with this addressOfOp has shared memory
    // space.
    if (addressOfOp.getGlobal(symbolTableCollection).getAddrSpace() == 3)
      addressOfOps.push_back(addressOfOp);
  });
  if (addressOfOps.size() == 0) return;
  OpBuilder builder(moduleOp);
  builder.setInsertionPoint(&moduleOp.front());
  auto type =
      LLVM::LLVMArrayType::get(IntegerType::get(builder.getContext(), 8), 0);
  LLVM::GlobalOp global = builder.create<LLVM::GlobalOp>(
      moduleOp.getLoc(), type, /*isConstant=*/false, LLVM::Linkage::External,
      "__dynamic_shared_memory__", Attribute(),
      /*alignment=*/16, /*addr_space=*/3);
  int64_t numberOfBytes = 0;
  llvm::SmallDenseMap<int64_t, int64_t> numberOfBytesPerGroup;
  // Replace the addressOfOps with correctly offseted pointers to dynamic
  // shared memory.
  llvm::SmallDenseMap<LLVM::GlobalOp, uint32_t> globalMemoryOffsetMap;
  for (auto addressOfOp : addressOfOps) {
    int64_t offset = 0;
    auto globalOp = addressOfOp.getGlobal(symbolTableCollection);
    if (globalMemoryOffsetMap.count(globalOp)) {
      offset = globalMemoryOffsetMap[globalOp];
    } else {
      int64_t *numberOfBytesGroup = &numberOfBytes;
      if (globalOp->hasAttr(kAllocAliasGroup)) {
        int64_t aliasGroup =
            globalOp->getAttr(kAllocAliasGroup).cast<IntegerAttr>().getInt();
        if (!numberOfBytesPerGroup.count(aliasGroup))
          numberOfBytesPerGroup[aliasGroup] = 0;
        numberOfBytesGroup = &numberOfBytesPerGroup[aliasGroup];
      }
      offset = *numberOfBytesGroup;
      if (std::optional<uint64_t> alignment = globalOp.getAlignment()) {
        offset = llvm::alignTo(offset, *alignment);
      }
      globalMemoryOffsetMap[globalOp] = offset;
      auto thisarray = globalOp.getType();
      DataLayout dataLayout = DataLayout::closest(addressOfOp);
      *numberOfBytesGroup =
          offset + dataLayout.getTypeSizeInBits(thisarray) / 8;
      numberOfBytes = std::max(numberOfBytes, *numberOfBytesGroup);
    }
    auto loc = addressOfOp.getLoc();
    builder.setInsertionPoint(addressOfOp);
    LLVM::AddressOfOp globalPtr =
        builder.create<LLVM::AddressOfOp>(loc, global);
    Value zero = builder.create<LLVM::ConstantOp>(
        loc, IntegerType::get(builder.getContext(), 64),
        builder.getI64IntegerAttr(0));
    Value offsetValue = builder.create<LLVM::ConstantOp>(
        loc, IntegerType::get(builder.getContext(), 64),
        builder.getI64IntegerAttr(offset));
    Value shiftedPtr = builder.create<LLVM::GEPOp>(
        loc, globalPtr.getType(), globalPtr, ValueRange({zero, offsetValue}));
    Value castPtr = builder.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(globalOp.getType(), global.getAddrSpace()),
        shiftedPtr);
    addressOfOp.replaceAllUsesWith(castPtr);
    addressOfOp.erase();
  }
  // Add the amount of shared memory required as an attribute.
  auto variantOp = moduleOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  if (variantOp != nullptr) {
    for (auto exportOp : variantOp.getOps<IREE::HAL::ExecutableExportOp>()) {
      exportOp->setAttr(exportOp.getWorkgroupLocalMemoryAttrName(),
                        builder.getIndexAttr(numberOfBytes));
    }
  }
}

namespace {

/// Scalarize math ops. It is needed to lower vector operation that don't have
/// vector support in CUDA and ROCM device library.
template <typename MathOpTy>
struct ScalarizeMathOp : public OpRewritePattern<MathOpTy> {
  using OpRewritePattern<MathOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(MathOpTy mathOp,
                                PatternRewriter &rewriter) const override {
    auto vecType = mathOp.getType().template dyn_cast<VectorType>();
    if (!vecType) return failure();
    Location loc = mathOp.getLoc();
    Value newVector = rewriter.create<arith::ConstantOp>(
        loc, vecType, rewriter.getZeroAttr(vecType));

    for (int64_t element : llvm::seq(int64_t(0), vecType.getNumElements())) {
      llvm::SmallVector<int64_t> indices;
      int64_t projectIndex = element;
      for (int64_t dim : llvm::seq(int64_t(0), vecType.getRank())) {
        int64_t index = projectIndex % vecType.getDimSize(dim);
        projectIndex = projectIndex / vecType.getDimSize(dim);
        indices.push_back(index);
      }
      SmallVector<Value> newOperands;
      for (Value operand : mathOp->getOperands()) {
        newOperands.push_back(
            rewriter.create<vector::ExtractOp>(loc, operand, indices));
      }
      Value scalarOp = rewriter.create<MathOpTy>(loc, newOperands);
      newVector =
          rewriter.create<vector::InsertOp>(loc, scalarOp, newVector, indices);
    }
    rewriter.replaceOp(mathOp, newVector);
    return success();
  }
};

struct ConvertSharedMemAllocOp : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    auto addressSpace = allocOp.getType()
                            .getMemorySpace()
                            .dyn_cast_or_null<gpu::AddressSpaceAttr>();
    if (!addressSpace ||
        addressSpace.getValue() != gpu::GPUDialect::getWorkgroupAddressSpace())
      return failure();
    ArrayRef<int64_t> shape = allocOp.getType().getShape();
    if (llvm::any_of(shape,
                     [](int64_t dim) { return dim == ShapedType::kDynamic; })) {
      return failure();
    }

    uint64_t alignement;
    if (llvm::Optional<uint64_t> alignementInfo = allocOp.getAlignment()) {
      alignement = alignementInfo.value();
    } else {
      // If no alignment specified align at least to the size of an element.
      Type elType = allocOp.getType().getElementType();
      if (auto shapeType = elType.dyn_cast<ShapedType>())
        alignement = shapeType.getSizeInBits() / 8;
      else
        alignement = elType.getIntOrFloatBitWidth() / 8;
    }
    // In CUDA workgroup memory is represented by a global variable.
    MemRefType allocType = allocOp.getType();
    auto funcOp = allocOp->getParentOfType<func::FuncOp>();
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    SymbolTable symbolTable(moduleOp);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(&moduleOp.front());
    auto global = rewriter.create<memref::GlobalOp>(
        funcOp.getLoc(), "__shared_memory__",
        /*sym_visibility=*/rewriter.getStringAttr("private"),
        /*type=*/allocType,
        /*initial_value=*/ElementsAttr(),
        /*constant=*/false,
        /*alignment=*/rewriter.getI64IntegerAttr(alignement));
    symbolTable.insert(global);
    if (allocOp->hasAttr(kAllocAliasGroup)) {
      global->setAttr(kAllocAliasGroup, allocOp->getAttr(kAllocAliasGroup));
    }

    rewriter.setInsertionPointToStart(&(*funcOp.getFunctionBody().begin()));
    rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(allocOp, global.getType(),
                                                     global.getName());
    return success();
  }
};

/// Returns the LLVM type of the global variable given the memref type `type`.
static Type convertGlobalMemrefTypeToLLVM(MemRefType type,
                                          LLVMTypeConverter &typeConverter) {
  // LLVM type for a global memref will be a multi-dimension array. For
  // declarations or uninitialized global memrefs, we can potentially flatten
  // this to a 1D array. However, for memref.global's with an initial value,
  // we do not intend to flatten the ElementsAttribute when going from std ->
  // LLVM dialect, so the LLVM type needs to me a multi-dimension array.
  Type elementType = typeConverter.convertType(type.getElementType());
  Type arrayTy = elementType;
  // Shape has the outermost dim at index 0, so need to walk it backwards
  for (int64_t dim : llvm::reverse(type.getShape()))
    arrayTy = LLVM::LLVMArrayType::get(arrayTy, dim);
  return arrayTy;
}

/// Alias group information needs to be propagated to LLVM dialect, therefore we
/// need our own lowering.
struct GlobalMemrefOpLowering
    : public ConvertOpToLLVMPattern<memref::GlobalOp> {
  using ConvertOpToLLVMPattern<memref::GlobalOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      memref::GlobalOp global, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    MemRefType type = global.getType();
    if (!isConvertibleAndHasIdentityMaps(type)) return failure();

    Type arrayTy = convertGlobalMemrefTypeToLLVM(type, *getTypeConverter());

    LLVM::Linkage linkage =
        global.isPublic() ? LLVM::Linkage::External : LLVM::Linkage::Private;

    Attribute initialValue = nullptr;
    if (!global.isExternal() && !global.isUninitialized()) {
      auto elementsAttr = global.getInitialValue()->cast<ElementsAttr>();
      initialValue = elementsAttr;

      // For scalar memrefs, the global variable created is of the element type,
      // so unpack the elements attribute to extract the value.
      if (type.getRank() == 0)
        initialValue = elementsAttr.getSplatValue<Attribute>();
    }

    uint64_t alignment = global.getAlignment().value_or(0);
    FailureOr<unsigned> addressSpace =
        getTypeConverter()->getMemRefAddressSpace(type);
    if (failed(addressSpace))
      return global.emitOpError(
          "memory space cannot be converted to an integer address space");
    auto newGlobal = rewriter.replaceOpWithNewOp<LLVM::GlobalOp>(
        global, arrayTy, global.getConstant(), linkage, global.getSymName(),
        initialValue, alignment, *addressSpace);
    if (!global.isExternal() && global.isUninitialized()) {
      Block *blk = new Block();
      newGlobal.getInitializerRegion().push_back(blk);
      rewriter.setInsertionPointToStart(blk);
      Value undef[] = {
          rewriter.create<LLVM::UndefOp>(global.getLoc(), arrayTy)};
      rewriter.create<LLVM::ReturnOp>(global.getLoc(), undef);
    }
    if (global->hasAttr(kAllocAliasGroup)) {
      newGlobal->setAttr(kAllocAliasGroup, global->getAttr(kAllocAliasGroup));
    }
    return success();
  }
};

/// Pass to test in dialect transformation used to legalize the IR before
/// convertToNVVM/ConvertToROCDL.
class TestLLVMGPULegalizeOpPass
    : public TestLLVMGPUScalarizeMathOpBase<TestLLVMGPULegalizeOpPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateScalarizeMathOps(patterns);
    populateConvertSharedMemoryAllocOps(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

using SetBinding = std::pair<APInt, APInt>;

/// Convention with the HAL side to pass kernel arguments.
/// The bindings are ordered based on binding set and binding index then
/// compressed and mapped to dense set of arguments.
/// This function looks at the symbols and return the mapping between
/// InterfaceBindingOp and kernel argument index.
/// For instance if the kernel has (set, bindings) A(0, 1), B(1, 5), C(0, 6) it
/// will return the mapping [A, 0], [C, 1], [B, 2]
static llvm::SmallDenseMap<SetBinding, size_t> getKernelArgMapping(
    Operation *funcOp) {
  llvm::SetVector<SetBinding> usedBindingSet;
  funcOp->walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
    usedBindingSet.insert(
        SetBinding(subspanOp.getSet(), subspanOp.getBinding()));
  });
  auto sparseBindings = usedBindingSet.takeVector();
  std::sort(sparseBindings.begin(), sparseBindings.end(),
            [](SetBinding lhs, SetBinding rhs) {
              if (lhs.first == rhs.first) return lhs.second.ult(rhs.second);
              return lhs.first.ult(rhs.first);
            });
  llvm::SmallDenseMap<SetBinding, size_t> mapBindingArgIndex;
  for (auto [index, binding] : llvm::enumerate(sparseBindings)) {
    mapBindingArgIndex[binding] = index;
  }
  return mapBindingArgIndex;
}

class ConvertFunc : public ConvertToLLVMPattern {
 public:
  explicit ConvertFunc(MLIRContext *context, LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(mlir::func::FuncOp::getOperationName(), context,
                             converter, 100) {}
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto funcOp = cast<func::FuncOp>(op);
    FunctionType fnType = funcOp.getFunctionType();
    (void)fnType;
    if (!funcOp.isPublic()) return failure();

    // illegal FuncOp must have 0 inputs.
    assert(fnType.getNumInputs() == 0 && fnType.getNumResults() == 0);

    TypeConverter::SignatureConversion signatureConverter(/*numOrigInputs=*/0);
    auto argMapping = getKernelArgMapping(funcOp);
    // There may be dead symbols, we pick i32 pointer as default argument type.
    SmallVector<Type, 8> llvmInputTypes(
        argMapping.size(), LLVM::LLVMPointerType::get(rewriter.getI32Type()));
    funcOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
      auto memrefType = subspanOp.getType().cast<MemRefType>();
      Type elType = memrefType.getElementType();
      auto llvmType = LLVM::LLVMPointerType::get(elType);
      llvmInputTypes[argMapping[SetBinding(subspanOp.getSet(),
                                           subspanOp.getBinding())]] = llvmType;
    });
    // As a convention with HAL, push constants are appended as kernel arguments
    // after all the binding inputs.
    uint64_t numConstants = 0;
    funcOp.walk([&](IREE::HAL::InterfaceConstantLoadOp constantOp) {
      numConstants =
          std::max(constantOp.getIndex().getZExtValue() + 1, numConstants);
    });
    llvmInputTypes.resize(argMapping.size() + numConstants,
                          rewriter.getI32Type());
    if (!llvmInputTypes.empty()) signatureConverter.addInputs(llvmInputTypes);

    // Construct newFunc with all attributes except return type & symbol name.
    SmallVector<NamedAttribute, 4> funcAttrs;
    for (auto attr : funcOp->getAttrs()) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == funcOp.getFunctionTypeAttrName()) {
        continue;
      }
      funcAttrs.push_back(attr);
    }

    auto llvmFuncType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(rewriter.getContext()), llvmInputTypes);
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        funcOp.getLoc(), funcOp.getName(), llvmFuncType,
        LLVM::Linkage::External, /*dso_local=*/false, /*cconv*/ LLVM::CConv::C,
        funcAttrs);

    // Copy all of funcOp's operations into newFuncOp's body and perform region
    // type conversion.
    rewriter.inlineRegionBefore(funcOp.getFunctionBody(),
                                newFuncOp.getFunctionBody(), newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getFunctionBody(),
                                           *typeConverter,
                                           &signatureConverter))) {
      return failure();
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

class ConvertIREEBindingSubspanOp : public ConvertToLLVMPattern {
 public:
  explicit ConvertIREEBindingSubspanOp(MLIRContext *context,
                                       LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(
            IREE::HAL::InterfaceBindingSubspanOp::getOperationName(), context,
            converter) {}

  /// Checks all subspanOps with the same binding has readonly attribute
  static bool checkAllSubspansReadonly(LLVM::LLVMFuncOp llvmFuncOp,
                                       APInt binding) {
    bool allReadOnly = false;
    llvmFuncOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp op) {
      if (op.getBinding() == binding) {
        if (!bitEnumContainsAny(op.getDescriptorFlags().value_or(
                                    IREE::HAL::DescriptorFlags::None),
                                IREE::HAL::DescriptorFlags::ReadOnly)) {
          allReadOnly = false;
          return WalkResult::interrupt();
        }
        allReadOnly = true;
      }
      return WalkResult::advance();
    });
    return allReadOnly;
  }

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Bail until nested under an LLVMFuncOp.
    auto llvmFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFuncOp) return failure();
    assert(llvmFuncOp.getNumArguments() > 0);

    auto argMapping = getKernelArgMapping(llvmFuncOp);
    Location loc = op->getLoc();
    auto subspanOp = cast<IREE::HAL::InterfaceBindingSubspanOp>(op);
    IREE::HAL::InterfaceBindingSubspanOpAdaptor adaptor(
        operands, op->getAttrDictionary());
    MemRefType memrefType =
        subspanOp.getResult().getType().dyn_cast<MemRefType>();
    mlir::BlockArgument llvmBufferArg = llvmFuncOp.getArgument(
        argMapping[SetBinding(subspanOp.getSet(), subspanOp.getBinding())]);
    // As a convention with HAL all the kernel argument pointers are 16Bytes
    // aligned.
    llvmFuncOp.setArgAttr(llvmBufferArg.getArgNumber(),
                          LLVM::LLVMDialect::getAlignAttrName(),
                          rewriter.getI32IntegerAttr(16));
    // It is safe to set the noalias attribute as it is guaranteed that the
    // ranges within bindings won't alias.
    llvmFuncOp.setArgAttr(llvmBufferArg.getArgNumber(),
                          LLVM::LLVMDialect::getNoAliasAttrName(),
                          rewriter.getUnitAttr());
    if (checkAllSubspansReadonly(llvmFuncOp, subspanOp.getBinding())) {
      // Setting the readonly attribute here will generate non-coherent cache
      // loads.
      llvmFuncOp.setArgAttr(llvmBufferArg.getArgNumber(),
                            LLVM::LLVMDialect::getReadonlyAttrName(),
                            rewriter.getUnitAttr());
    }
    // Add the byte offset.
    Value llvmBufferBasei8Ptr = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(rewriter.getIntegerType(8),
                                   llvmBufferArg.getType()
                                       .cast<LLVM::LLVMPointerType>()
                                       .getAddressSpace()),
        llvmBufferArg);
    if (adaptor.getByteOffset()) {
      llvmBufferBasei8Ptr = rewriter.create<LLVM::GEPOp>(
          loc, llvmBufferBasei8Ptr.getType(), llvmBufferBasei8Ptr,
          adaptor.getByteOffset());
    }
    auto llvmPtrType = LLVM::LLVMPointerType::get(memrefType.getElementType());
    Value llvmBufferBasePtr =
        rewriter.create<LLVM::BitcastOp>(loc, llvmPtrType, llvmBufferBasei8Ptr);
    if (memrefType.hasStaticShape()) {
      auto desc = MemRefDescriptor::fromStaticShape(
          rewriter, loc, *getTypeConverter(), memrefType, llvmBufferBasePtr);
      rewriter.replaceOp(op, {desc});
    } else {
      ValueRange dynamicDims = adaptor.getDynamicDims();
      assert(memrefType.getNumDynamicDims() == dynamicDims.size());
      int64_t rank = memrefType.getRank();

      // Build MemRef descriptor for this interface binding.
      auto desc = MemRefDescriptor::undef(
          rewriter, loc, typeConverter->convertType(memrefType));
      desc.setAllocatedPtr(rewriter, loc, llvmBufferBasePtr);
      desc.setAlignedPtr(rewriter, loc, llvmBufferBasePtr);
      desc.setConstantOffset(rewriter, loc, 0);

      // Update memref descriptor shape. Dynamic dimensions can be mixed with
      // static dimensions, like [128, ?, 128].
      int dynamicDimIndex = 0;
      for (int i = 0; i < rank; ++i) {
        if (memrefType.isDynamicDim(i)) {
          desc.setSize(rewriter, loc, i, dynamicDims[dynamicDimIndex++]);
        } else {
          desc.setConstantSize(rewriter, loc, i, memrefType.getDimSize(i));
        }
      }

      // Compute and update strides. Assume that MemRefs are row-major, that is,
      // following index linearization:
      //   x[i, j, k] = i * x.dim[1] * x.dim[2] + j * x.dim[2] + k
      desc.setConstantStride(rewriter, loc, rank - 1, 1);
      for (int i = rank - 2; i >= 0; --i) {
        auto stride = desc.stride(rewriter, loc, i + 1);
        auto dim = desc.size(rewriter, loc, i + 1);
        Value strideVal = rewriter.create<LLVM::MulOp>(loc, stride, dim);
        desc.setStride(rewriter, loc, i, strideVal);
      }
      rewriter.replaceOp(op, {desc});
    }

    return success();
  }
};

class ConvertIREEConstantOp : public ConvertToLLVMPattern {
 public:
  explicit ConvertIREEConstantOp(MLIRContext *context,
                                 LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(
            IREE::HAL::InterfaceConstantLoadOp::getOperationName(), context,
            converter) {}
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Bail until nested under an LLVMFuncOp.
    auto llvmFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFuncOp) return failure();
    assert(llvmFuncOp.getNumArguments() > 0);

    auto argMapping = getKernelArgMapping(llvmFuncOp);
    auto ireeConstantOp = cast<IREE::HAL::InterfaceConstantLoadOp>(op);
    mlir::BlockArgument llvmBufferArg = llvmFuncOp.getArgument(
        argMapping.size() + ireeConstantOp.getIndex().getZExtValue());
    assert(llvmBufferArg.getType().isInteger(32));
    Type dstType = getTypeConverter()->convertType(ireeConstantOp.getType());
    rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(op, dstType, llvmBufferArg);
    return success();
  }
};

/// A pattern to convert hal.interface.workgroup.id/count/size into
/// corresponding GPU ops.
template <typename InterfaceOpTy, typename NewOpTy>
struct HALInterfaceWorkgroupOpsConverter final
    : public OpConversionPattern<InterfaceOpTy> {
  using OpConversionPattern<InterfaceOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      InterfaceOpTy op, typename InterfaceOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    int32_t index = static_cast<int32_t>(op.getDimension().getSExtValue());
    std::array<gpu::Dimension, 3> dimAttr{gpu::Dimension::x, gpu::Dimension::y,
                                          gpu::Dimension::z};
    rewriter.replaceOpWithNewOp<NewOpTy>(op, op.getType(), dimAttr[index]);
    return success();
  }
};

}  // anonymous namespace

void populateLLVMConversionPatterns(MLIRContext *context,
                                    RewritePatternSet &patterns,
                                    LLVMTypeConverter &converter) {
  patterns.add<GlobalMemrefOpLowering>(converter, 2);
  patterns
      .insert<ConvertFunc, ConvertIREEBindingSubspanOp, ConvertIREEConstantOp>(
          context, converter);
}

void populateScalarizeMathOps(RewritePatternSet &patterns) {
  patterns.add<ScalarizeMathOp<math::SqrtOp>, ScalarizeMathOp<math::AbsFOp>,
               ScalarizeMathOp<math::AtanOp>, ScalarizeMathOp<math::Atan2Op>,
               ScalarizeMathOp<math::CeilOp>, ScalarizeMathOp<math::CosOp>,
               ScalarizeMathOp<math::ExpOp>, ScalarizeMathOp<math::ExpM1Op>,
               ScalarizeMathOp<math::FloorOp>, ScalarizeMathOp<math::LogOp>,
               ScalarizeMathOp<math::Log1pOp>, ScalarizeMathOp<math::Log10Op>,
               ScalarizeMathOp<math::Log2Op>, ScalarizeMathOp<math::PowFOp>,
               ScalarizeMathOp<math::RsqrtOp>, ScalarizeMathOp<math::SinOp>,
               ScalarizeMathOp<math::SqrtOp>, ScalarizeMathOp<math::TanhOp>>(
      patterns.getContext());
}

void populateConvertSharedMemoryAllocOps(RewritePatternSet &patterns) {
  patterns.add<ConvertSharedMemAllocOp>(patterns.getContext());
}

void populateLowerHALInterfaceOp(RewritePatternSet &patterns) {
  patterns.insert<HALInterfaceWorkgroupOpsConverter<
                      IREE::HAL::InterfaceWorkgroupIDOp, gpu::BlockIdOp>,
                  HALInterfaceWorkgroupOpsConverter<
                      IREE::HAL::InterfaceWorkgroupCountOp, gpu::GridDimOp>>(
      patterns.getContext());
}

std::unique_ptr<OperationPass<ModuleOp>> createTestLLVMGPULegalizePass() {
  return std::make_unique<TestLLVMGPULegalizeOpPass>();
}

static IntegerAttr wrapNumericMemorySpace(MLIRContext *ctx, unsigned space) {
  return IntegerAttr::get(IntegerType::get(ctx, 64), space);
}

void populateGpuMemorySpaceAttributeConversions(
    TypeConverter &typeConverter, const MemorySpaceMapping &mapping) {
  typeConverter.addTypeAttributeConversion(
      [mapping](BaseMemRefType type, gpu::AddressSpaceAttr memorySpaceAttr) {
        gpu::AddressSpace memorySpace = memorySpaceAttr.getValue();
        unsigned addressSpace = mapping(memorySpace);
        return wrapNumericMemorySpace(memorySpaceAttr.getContext(),
                                      addressSpace);
      });
}

}  // namespace iree_compiler
}  // namespace mlir
