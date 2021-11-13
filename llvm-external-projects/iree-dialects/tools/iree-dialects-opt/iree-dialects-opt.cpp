// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/PyDM/IR/PyDMDialect.h"
#include "iree-dialects/Dialect/PyDM/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::Input;

int main(int argc, char **argv) {
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();

  registerTransformsPasses();
  registerSCFPasses();

  // Local dialects.
  mlir::iree_compiler::IREE::PYDM::registerPasses();

  DialectRegistry registry;
  registry.insert<
      // Local dialects
      mlir::iree_compiler::IREE::Input::IREEInputDialect,
      mlir::iree_compiler::IREE::PYDM::IREEPyDMDialect,
      // Upstream dialects
      mlir::StandardOpsDialect, mlir::scf::SCFDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR modular optimizer driver\n", registry,
                        /*preloadDialectsInContext=*/false));
}
