// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

// Dialect specific
#ifdef IREE_HAVE_MHLO_INPUT
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#include "mhlo/IR/hlo_ops.h"
#endif
#ifdef IREE_HAVE_TOSA_INPUT
#include "iree/compiler/InputConversion/TOSA/Passes.h"
#endif  // IREE_HAVE_TOSA_INPUT
#ifdef IREE_HAVE_TORCH_INPUT
#include "iree/compiler/InputConversion/TMTensor/Passes.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#endif  // IREE_HAVE_TORCH_INPUT

namespace mlir::iree_compiler {

struct InputConversionPipelinePass
    : public InputConversionPipelineBase<InputConversionPipelinePass> {
  void runOnOperation() override;
};

#if defined(IREE_HAVE_MHLO_INPUT) || defined(IREE_HAVE_TOSA_INPUT)

// All the features seen that should be handled during input conversion.
struct InputFeatures {
  // MHLO features.
  bool hasMHLO = false;
  bool hasStableHLO = false;
  // - XLA import features.
  bool hasTuples = false;

  // TOSA features.
  bool hasTOSA = false;

  // tm_tensor
  bool hasTmTensor = false;
};

static void populateHloFeatures(Operation* op, InputFeatures& features) {
  features.hasMHLO = true;
  if (features.hasTuples) return;

  if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
    FunctionType type = dyn_cast<FunctionType>(funcOp.getFunctionType());
    for (auto t : type.getResults()) {
      if (isa<TupleType>(t)) return (void)(features.hasTuples = true);
    }
    for (auto t : type.getInputs()) {
      if (isa<TupleType>(t)) return (void)(features.hasTuples = true);
    }
  }

  // Check for tuple operands or results.
  for (auto t : op->getOperandTypes())
    if (isa<TupleType>(t)) return (void)(features.hasTuples = true);
  for (auto t : op->getResultTypes())
    if (isa<TupleType>(t)) return (void)(features.hasTuples = true);
}

static void populateFeatures(Operation* op, const Dialect* mhloDialect,
                             const Dialect* tmTensorDialect,
                             const Dialect* tosaDialect,
                             InputFeatures& features) {
  Dialect* d = op->getDialect();
  if (d == mhloDialect) return populateHloFeatures(op, features);
  if (d == tosaDialect) return (void)(features.hasTOSA = true);
  if (d == tmTensorDialect) return (void)(features.hasTmTensor = true);
}

void InputConversionPipelinePass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext* ctxt = &getContext();

  InputFeatures features;

  const Dialect* mhloDialect = nullptr;
  const Dialect* tosaDialect = nullptr;
  const Dialect* tmTensorDialect = nullptr;
#ifdef IREE_HAVE_MHLO_INPUT
  mhloDialect = ctxt->getLoadedDialect<mhlo::MhloDialect>();
#endif
#ifdef IREE_HAVE_TORCH_INPUT
  tmTensorDialect = ctxt->getLoadedDialect<torch::TMTensor::TMTensorDialect>();
#endif
#ifdef IREE_HAVE_TOSA_INPUT
  tosaDialect = ctxt->getLoadedDialect<tosa::TosaDialect>();
#endif

  auto res = module.walk([&](Operation* op) {
    populateFeatures(op, mhloDialect, tmTensorDialect, tosaDialect, features);
    if (features.hasMHLO && features.hasTOSA) {
      module.emitError("not yet implemented mixture of *HLO and TOSA");
      return WalkResult::interrupt();
    }
    if (features.hasMHLO && features.hasTmTensor) {
      module.emitError("not yet implemented mixture of *HLO and TM Tensor");
      return WalkResult::interrupt();
    }
    if (features.hasTOSA && features.hasTmTensor) {
      module.emitError("not yet implemented mixture of TOSA and TM Tensor");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();

  OpPassManager pm(ModuleOp::getOperationName(),
                   OpPassManager::Nesting::Explicit);
  if (features.hasMHLO) {
    if (features.hasTuples)
      MHLO::buildXLAInputConversionPassPipeline(pm);
    else
      MHLO::buildMHLOInputConversionPassPipeline(pm);
  }
  if (features.hasTOSA) buildTOSAInputConversionPassPipeline(pm);
  if (features.hasTmTensor) {
    pm.addNestedPass<func::FuncOp>(
        TMTensor::createConvertTMTensorToLinalgExtPass());
  }

  if (failed(runPipeline(pm, module))) signalPassFailure();
}

#else
void InputConversionPipelinePass::runOnOperation() {}
#endif

std::unique_ptr<OperationPass<ModuleOp>> createInputConversionPipelinePass() {
  return std::make_unique<InputConversionPipelinePass>();
}

}  // namespace mlir::iree_compiler
