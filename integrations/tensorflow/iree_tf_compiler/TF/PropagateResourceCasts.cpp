// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/TF/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace iree_integrations {
namespace TF {

// Determine whether we should bypass the cast for input (a) to output (b).
static bool shouldBypassCast(ShapedType a, ShapedType b) {
  // If the element type changes the cast is required.
  if (a.getElementType() != b.getElementType()) {
    return false;
  }

  // If we have no rank for the output we should bypass the cast.
  if (!b.hasRank()) {
    return true;
  }

  // If the input doesn't have a rank we can't gain informatio
  if (!a.hasRank()) {
    return false;
  }

  if (a.getRank() != b.getRank()) {
    return false;
  }

  auto a_shape = a.getShape();
  auto b_shape = b.getShape();
  for (auto pair : llvm::zip(a_shape, b_shape)) {
    auto a_dim = std::get<0>(pair);
    auto b_dim = std::get<1>(pair);
    if (a_dim != b_dim && a_dim == -1) {
      return false;
    }
  }
  return true;
}

// Attempts to propagate resource casts by bypassing them when they are not
// necessary or can further refine required types.
class PropagateResourceCastsPass
    : public PassWrapper<PropagateResourceCastsPass, OperationPass<ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::TF::TensorFlowDialect>();
  }

  StringRef getArgument() const override {
    return "iree-tf-propagate-resource-casts";
  }

  StringRef getDescription() const override {
    return "Propagates tf.resource type casts";
  }

  void runOnOperation() override {
    auto operation = getOperation();
    for (auto func : operation.getOps<func::FuncOp>()) {
      for (auto cast : func.getOps<mlir::TF::CastOp>()) {
        auto input = cast.getX();
        auto output = cast.getResult();

        auto inputTy = input.getType().cast<ShapedType>();
        auto outputTy = output.getType().cast<ShapedType>();

        // If the input/output types match we can just bypass it.
        if (inputTy == outputTy) {
          output.replaceAllUsesWith(input);
          continue;
        }

        auto inputElementTy =
            inputTy.getElementType().dyn_cast<mlir::TF::ResourceType>();
        auto outputElementTy =
            outputTy.getElementType().dyn_cast<mlir::TF::ResourceType>();

        // Check whether it is a
        if (!inputElementTy || !outputElementTy ||
            inputElementTy.getSubtypes().empty()) {
          continue;
        }

        auto input_resource_ty = inputElementTy.getSubtypes().front();
        if (!outputElementTy.getSubtypes().empty()) {
          auto output_resource_ty = outputElementTy.getSubtypes().front();
          if (!shouldBypassCast(input_resource_ty, output_resource_ty)) {
            continue;
          }
        }

        // TODO(suderman): Check which functions could be updated and
        // substitute.
        output.replaceAllUsesWith(input);
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createPropagateResourceCastsPass() {
  return std::make_unique<PropagateResourceCastsPass>();
}

static PassRegistration<PropagateResourceCastsPass> pass;

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir
