// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_IR_HALDIALECT_H_
#define IREE_COMPILER_DIALECT_HAL_IR_HALDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class HALDialect : public Dialect {
 public:
  explicit HALDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "hal"; }

  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;
  void printAttribute(Attribute attr, DialectAsmPrinter &p) const override;

  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &p) const override;

  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;

 private:
  /// Register the attributes of this dialect.
  void registerAttributes();
  /// Register the types of this dialect.
  void registerTypes();
};

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_IR_HALDIALECT_H_
