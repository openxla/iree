// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_RESOLVEDEVICEPROMISESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-resolve-device-promises
//===----------------------------------------------------------------------===//

struct ResolveDevicePromisesPass
    : public IREE::HAL::impl::ResolveDevicePromisesPassBase<
          ResolveDevicePromisesPass> {
  using IREE::HAL::impl::ResolveDevicePromisesPassBase<
      ResolveDevicePromisesPass>::ResolveDevicePromisesPassBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Resolves a #hal.device.promise attr to a #hal.device.affinity. Fails if
    // the referenced device is not found.
    SymbolTable symbolTable(moduleOp);
    auto resolvePromise = [&](Operation *fromOp,
                              IREE::HAL::DevicePromiseAttr promiseAttr)
        -> FailureOr<IREE::Stream::AffinityAttr> {
      auto deviceOp =
          symbolTable.lookupNearestSymbolFrom<IREE::Util::GlobalOpInterface>(
              fromOp, promiseAttr.getDevice());
      if (!deviceOp) {
        return fromOp->emitOpError()
               << "references a promised device that was not declared: "
               << promiseAttr;
      }
      return IREE::HAL::DeviceAffinityAttr::get(
          &getContext(), FlatSymbolRefAttr::get(deviceOp),
          promiseAttr.getQueueMask());
    };

    // Resolves any #hal.device.promise attr on the op.
    auto affinityName = StringAttr::get(&getContext(), "stream.affinity");
    auto resolvePromisesOnOp = [&](Operation *op) -> WalkResult {
      if (auto affinityOp = dyn_cast<IREE::Stream::AffinityOpInterface>(op)) {
        if (auto promiseAttr =
                dyn_cast_if_present<IREE::HAL::DevicePromiseAttr>(
                    affinityOp.getAffinity())) {
          auto resolvedAttrOr = resolvePromise(op, promiseAttr);
          if (failed(resolvedAttrOr))
            return WalkResult::interrupt();
          affinityOp.setAffinity(resolvedAttrOr.value());
        }
      } else if (auto promiseAttr =
                     op->getAttrOfType<IREE::HAL::DevicePromiseAttr>(
                         affinityName)) {
        auto resolvedAttrOr = resolvePromise(op, promiseAttr);
        if (failed(resolvedAttrOr))
          return WalkResult::interrupt();
        op->setAttr(affinityName, resolvedAttrOr.value());
      }
      return WalkResult::advance();
    };

    // Walk the entire module and replace promises.
    // We skip any symbol table op as all devices are top-level only.
    if (resolvePromisesOnOp(moduleOp).wasInterrupted())
      return signalPassFailure();
    if (moduleOp
            .walk([&](Operation *op) {
              if (op->hasTrait<OpTrait::SymbolTable>())
                return WalkResult::skip();
              return resolvePromisesOnOp(op);
            })
            .wasInterrupted()) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
