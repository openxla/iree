// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_HAL_INTERFACES
#define IREE_DIALECT_HAL_INTERFACES

include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// IREE::HAL::DeviceInitializationAttrInterface
//===----------------------------------------------------------------------===//

def HAL_DeviceInitializationAttrInterface :
    AttrInterface<"DeviceInitializationAttrInterface", [
      TypedAttrInterface,
    ]> {
  let description = [{
    Interface for attributes controlling device initialization.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        prints a string description of the initialization specification for
        inclusion in error messages. May include internal newlines but no
        newline is expected at the end.
      }],
      /*retTy=*/"void",
      /*methodName=*/"printStatusDescription",
      /*args=*/(ins "llvm::raw_ostream &":$os),
      /*methodBody=*/[{}]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Builds a `util.initializer` body responsible for initializing a device
        global. Returns the device value that should be stored into the global.
        The name provided is an informal identifier that can be used to produce
        user-level error messages that reference the device.
      }],
      /*retTy=*/"Value",
      /*methodName=*/"buildDeviceEnumeration",
      /*args=*/(ins
        "Location":$loc,
        "const IREE::HAL::TargetRegistry &":$targetRegistry,
        "OpBuilder &":$builder
      ),
      /*methodBody=*/[{}]
    >,
  ];
}

#endif  // IREE_DIALECT_HAL_INTERFACES
