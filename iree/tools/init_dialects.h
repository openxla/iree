// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This files defines a helper to trigger the registration of dialects and
// passes to the system.
//
// Based on MLIR's InitAllDialects but without dialects we don't care about.

#ifndef MLIR_INIT_DIALECTS_H_
#define MLIR_INIT_DIALECTS_H_

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/FxpMathOps/FxpMathOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/QuantOps/QuantOps.h"
#include "mlir/Dialect/SDBM/SDBMDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/VectorOps/VectorOps.h"
#include "mlir/IR/Dialect.h"

namespace mlir {

// This function should be called before creating any MLIRContext if one expect
// all the possible dialects to be made available to the context automatically.
inline void registerMlirDialects() {
  static bool init_once = []() {
    registerDialect<AffineOpsDialect>();
    registerDialect<fxpmath::FxpMathOpsDialect>();
    registerDialect<gpu::GPUDialect>();
    registerDialect<LLVM::LLVMDialect>();
    registerDialect<linalg::LinalgDialect>();
    registerDialect<loop::LoopOpsDialect>();
    registerDialect<quant::QuantizationDialect>();
    registerDialect<spirv::SPIRVDialect>();
    registerDialect<StandardOpsDialect>();
    registerDialect<vector::VectorOpsDialect>();
    registerDialect<SDBMDialect>();
    return true;
  }();
  (void)init_once;
}
}  // namespace mlir

#endif  // MLIR_INIT_DIALECTS_H_
