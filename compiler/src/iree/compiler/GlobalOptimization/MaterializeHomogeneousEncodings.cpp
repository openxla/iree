// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::GlobalOptimization {

using FunctionLikeNest =
    MultiOpNest<IREE::Util::InitializerOp, IREE::Util::FuncOp>;

// Returns a list of target devices that may be active for the given
// operation. This will recursively walk parent operations until one with
// the `hal.device.targets` attribute is found.
static SmallVector<IREE::HAL::DeviceTargetAttr>
lookupDeviceTargetAttrs(Operation *op) {
  auto attrId = mlir::StringAttr::get(op->getContext(), "hal.device.targets");
  while (op) {
    auto targetsAttr = op->getAttrOfType<ArrayAttr>(attrId);
    if (targetsAttr) {
      SmallVector<IREE::HAL::DeviceTargetAttr> result;
      for (auto targetAttr : targetsAttr) {
        result.push_back(llvm::cast<IREE::HAL::DeviceTargetAttr>(targetAttr));
      }
      return result;
    }
    op = op->getParentOp();
  }
  return {}; // No devices found; let caller decide what to do.
}

static SmallVector<IREE::HAL::ExecutableTargetAttr, 4>
lookupExecutableTargets(Operation *op) {
  SmallVector<IREE::HAL::ExecutableTargetAttr> resultAttrs;
  for (auto deviceTargetAttr : lookupDeviceTargetAttrs(op)) {
    for (auto executableTargetAttr : deviceTargetAttr.getExecutableTargets()) {
      if (!llvm::is_contained(resultAttrs, executableTargetAttr)) {
        resultAttrs.push_back(executableTargetAttr);
      }
    }
  }
  return resultAttrs;
}

class MaterializeHomogeneousEncodingsPass
    : public MaterializeHomogeneousEncodingsBase<
          MaterializeHomogeneousEncodingsPass> {
public:
  MaterializeHomogeneousEncodingsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
  }

  void runNopPipeline(ModuleOp &moduleOp) {
    OpPassManager passManager(moduleOp.getOperationName());
    FunctionLikeNest(passManager).addPass(createMaterializeEncodingIntoNopPass);
    FunctionLikeNest(passManager).addPass(createCanonicalizerPass);
    if (failed(runPipeline(passManager, moduleOp))) {
      return signalPassFailure();
    }
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto executableTargets = lookupExecutableTargets(moduleOp);
    if (executableTargets.size() != 1) {
      return runNopPipeline(moduleOp);
    }
    // TODO: vmvx has its own logic about supporting dynamic tile
    // sizes. It is not fully integrated into the pipeline, so we remain the
    // materialization to the end.
    auto executableTarget = executableTargets[0];
    if (executableTarget.getBackend() == "vmvx") {
      return;
    }

    // Only llvm-cpu backends handle encodings for now, others just go with nop.
    if (executableTarget.getBackend() != "llvm-cpu") {
      return runNopPipeline(moduleOp);
    }

    OpPassManager passManager(moduleOp.getOperationName());
    passManager.addPass(createCPUMaterializeUpperBoundTileSizePass());
    passManager.addPass(createCPUMaterializeEncodingPass());
    if (failed(runPipeline(passManager, moduleOp))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
createMaterializeHomogeneousEncodingsPass() {
  return std::make_unique<MaterializeHomogeneousEncodingsPass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
