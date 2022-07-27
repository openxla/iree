// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"

#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Parser/Parser.h"

namespace mlir {
namespace iree_compiler {

// TODO(benvanik): replace with iree/compiler/Utils/ModuleUtils.h.
// There may be some special insertion order arrangement required based on the
// nested vm.module here.

LogicalResult appendImportModule(IREE::VM::ModuleOp importModuleOp,
                                 ModuleOp targetModuleOp) {
  SymbolTable symbolTable(targetModuleOp);
  OpBuilder targetBuilder(targetModuleOp);
  targetBuilder.setInsertionPoint(&targetModuleOp.getBody()->back());
  importModuleOp.walk([&](IREE::VM::ImportOp importOp) {
    std::string fullName =
        (importModuleOp.getName() + "." + importOp.getName()).str();
    auto *existingOp = symbolTable.lookup(fullName);
    // TODO(benvanik): verify that the imports match.
    if (!existingOp) {
      auto clonedOp = cast<IREE::VM::ImportOp>(targetBuilder.clone(*importOp));
      mlir::StringAttr fullNameAttr =
          mlir::StringAttr::get(clonedOp.getContext(), fullName);
      clonedOp.setName(fullNameAttr);
      clonedOp.setPrivate();
    }
  });
  return success();
}

LogicalResult appendImportModule(StringRef importModuleSrc,
                                 ModuleOp targetModuleOp) {
  auto importModuleRef = mlir::parseSourceString<mlir::ModuleOp>(
      importModuleSrc, targetModuleOp.getContext());
  if (!importModuleRef) {
    return targetModuleOp.emitError()
           << "unable to append import module; import module failed to parse";
  }
  for (auto importModuleOp : importModuleRef->getOps<IREE::VM::ModuleOp>()) {
    if (failed(appendImportModule(importModuleOp, targetModuleOp))) {
      importModuleOp.emitError() << "failed to import module";
    }
  }
  return success();
}

Value castToImportType(Value value, Type targetType,
                       ConversionPatternRewriter &rewriter) {
  auto sourceType = value.getType();
  if (sourceType == targetType) return value;
  bool sourceIsInteger = sourceType.isa<IntegerType>();

  // Allow bitcast between same width float/int types. This is used for
  // marshalling to "untyped" VM interfaces, which will have an integer type.
  if (sourceType.isa<FloatType>() && targetType.isa<IntegerType>() &&
      sourceType.getIntOrFloatBitWidth() ==
          targetType.getIntOrFloatBitWidth()) {
    return rewriter.create<mlir::arith::BitcastOp>(value.getLoc(), targetType,
                                                   value);
  } else if (sourceIsInteger &&
             (targetType.isSignedInteger() || targetType.isSignlessInteger())) {
    if (targetType.getIntOrFloatBitWidth() >
        sourceType.getIntOrFloatBitWidth()) {
      return rewriter.create<mlir::arith::ExtSIOp>(value.getLoc(), targetType,
                                                   value);
    } else {
      return rewriter.create<mlir::arith::TruncIOp>(value.getLoc(), targetType,
                                                    value);
    }
  } else if (sourceIsInteger && targetType.isUnsignedInteger()) {
    if (targetType.getIntOrFloatBitWidth() >
        sourceType.getIntOrFloatBitWidth()) {
      return rewriter.create<mlir::arith::ExtUIOp>(value.getLoc(), targetType,
                                                   value);
    } else {
      return rewriter.create<mlir::arith::TruncIOp>(value.getLoc(), targetType,
                                                    value);
    }
  } else {
    return value;
  }
}

Value castFromImportType(Value value, Type targetType,
                         ConversionPatternRewriter &rewriter) {
  // Right now the to-import and from-import types are the same.
  return castToImportType(value, targetType, rewriter);
}

void copyImportAttrs(IREE::VM::ImportOp importOp, Operation *callOp) {
  if (importOp->hasAttr("nosideeffects")) {
    callOp->setAttr("nosideeffects", UnitAttr::get(importOp.getContext()));
  }
}

namespace detail {

// Makes a human-readable symbol name for the given string value.
// This is not uniqued and may need uniquing before being added to the symbol
// table.
//
// For example:
//   'Some string!' -> '_utf8_some_string'
//   'I'm a really long'... -> '_utf8_im_a_really_long'
static std::string makeSafeIdentifier(StringRef unsafeIdentifier) {
  std::string result = "_utf8_";
  llvm::raw_string_ostream os(result);
  bool lastUnderscore = true;
  for (char c : unsafeIdentifier) {
    if (!llvm::isPrint(c)) continue;
    if (llvm::isAlnum(c)) {
      os << llvm::toLower(c);
      lastUnderscore = false;
    } else if (!lastUnderscore) {
      os << "_";
      lastUnderscore = true;
    }
  }
  std::string prefix = os.str().substr(0, 32);
  if (!StringRef(prefix).endswith("_")) {
    prefix += "_";
  }
  return prefix + llvm::utohexstr(static_cast<uint64_t>(
                      llvm::hash_value(unsafeIdentifier)));
}

size_t getSegmentSpanSize(Type spanType) {
  if (auto tupleType = spanType.dyn_cast<TupleType>()) {
    return tupleType.size();
  } else {
    return 1;
  }
}

}  // namespace detail

Value createStringTableValue(Location loc, StringAttr attrValue, Type inputType,
                             OpBuilder &builder) {
  auto stringValue = attrValue.getValue();

  // Make an identifier-friendly version of the string so that the value is
  // more readable in IR (so "I'm some string" becomes "im_some_string", etc).
  auto safeIdentifier = detail::makeSafeIdentifier(stringValue);

  // Encode the string value bytes into an elements attr as UTF-8 bytes.
  SmallVector<APInt, 16> stringBytes(stringValue.size());
  for (int i = 0; i < stringValue.size(); ++i) {
    stringBytes[i] = APInt(8, stringValue[i]);
  }
  auto utf8Bytes = DenseIntElementsAttr::get(
      VectorType::get({static_cast<int64_t>(stringBytes.size())},
                      builder.getIntegerType(8)),
      stringBytes);

  // Embed the UTF-8 bytes as a vm.rodata blob.
  return builder.create<IREE::VM::RodataInlineOp>(
      loc,
      IREE::VM::RefType::get(IREE::VM::BufferType::get(builder.getContext())),
      builder.getStringAttr(safeIdentifier), utf8Bytes,
      /*alignment=*/builder.getI64IntegerAttr(1));
}

namespace detail {

Optional<SmallVector<Value, 4>> rewriteAttrToOperands(
    Location loc, Attribute attrValue, Type inputType,
    ConversionPatternRewriter &rewriter) {
  if (auto intAttr = attrValue.dyn_cast<IntegerAttr>()) {
    // NOTE: we intentionally go to std.constant ops so that the standard
    // conversions can do their job. If we want to remove the dependency
    // from standard ops in the future we could instead go directly to
    // one of the vm constant ops.
    auto constValue = rewriter.createOrFold<mlir::arith::ConstantOp>(
        loc, inputType,
        IntegerAttr::get(inputType,
                         APInt(32, static_cast<int32_t>(intAttr.getInt()))));
    return {{constValue}};
  } else if (auto elementsAttr = attrValue.dyn_cast<DenseIntElementsAttr>()) {
    SmallVector<Value, 4> elementValues;
    elementValues.reserve(elementsAttr.getNumElements());
    for (auto intAttr : elementsAttr.getValues<Attribute>()) {
      elementValues.push_back(rewriter.createOrFold<mlir::arith::ConstantOp>(
          loc, elementsAttr.getType().getElementType(), intAttr));
    }
    return elementValues;
  } else if (auto arrayAttr = attrValue.dyn_cast<ArrayAttr>()) {
    SmallVector<Value, 4> allValues;
    for (auto elementAttr : arrayAttr) {
      auto flattenedValues =
          rewriteAttrToOperands(loc, elementAttr, inputType, rewriter);
      if (!flattenedValues) return llvm::None;
      allValues.append(flattenedValues->begin(), flattenedValues->end());
    }
    return allValues;
  } else if (auto strAttr = attrValue.dyn_cast<StringAttr>()) {
    return {{createStringTableValue(loc, strAttr, inputType, rewriter)}};
  }

  // This may be a custom dialect type. As we can't trivially access the storage
  // of these we need to ask the dialect to do it for us.
  auto *conversionInterface =
      attrValue.getDialect()
          .getRegisteredInterface<VMConversionDialectInterface>();
  if (conversionInterface) {
    bool anyFailed = false;
    SmallVector<Value, 4> allValues;
    if (auto tupleType = inputType.dyn_cast<TupleType>()) {
      // Custom dialect type maps into a tuple; we expect 1:1 tuple elements to
      // attribute storage elements.
      auto tupleTypes = llvm::to_vector<4>(tupleType.getTypes());
      int ordinal = 0;
      conversionInterface->walkAttributeStorage(
          attrValue, [&](Attribute elementAttr) {
            if (anyFailed) return;
            auto elementType = tupleTypes[ordinal++];
            auto flattenedValues =
                rewriteAttrToOperands(loc, elementAttr, elementType, rewriter);
            if (!flattenedValues) {
              anyFailed = true;
              return;
            }
            allValues.append(flattenedValues->begin(), flattenedValues->end());
          });
    } else {
      // Custom dialect type maps into zero or more input types (ala arrays).
      conversionInterface->walkAttributeStorage(
          attrValue, [&](Attribute elementAttr) {
            if (anyFailed) return;
            auto flattenedValues =
                rewriteAttrToOperands(loc, elementAttr, inputType, rewriter);
            if (!flattenedValues) {
              anyFailed = true;
              return;
            }
            allValues.append(flattenedValues->begin(), flattenedValues->end());
          });
    }
    if (anyFailed) return llvm::None;
    return allValues;
  }

  emitError(loc) << "unsupported attribute encoding: " << attrValue.getType();
  return llvm::None;
}

}  // namespace detail

}  // namespace iree_compiler
}  // namespace mlir
