// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_VECTOREXT_IR_VECTOREXTOPS_H_
#define IREE_DIALECTS_DIALECT_VECTOREXT_IR_VECTOREXTOPS_H_

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtInterfaces.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// clang-format off

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtEnums.h.inc" // IWYU pragma: export

#define GET_ATTRDEF_CLASSES
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtAttrs.h.inc" // IWYU pragma: export

#define GET_OP_CLASSES
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h.inc" // IWYU pragma: export

// clang-format on

namespace mlir::iree_compiler::IREE::VectorExt {

/// Dimensional Strided Iterator class used to represent
/// an iterator through a single dimension of the layout.
class DimensionalIterator {
public:
  DimensionalIterator(int64_t position = 0, int64_t stride = 1)
      : position(position), stride(stride) {}
  int64_t operator*() const { return position; }
  DimensionalIterator &operator++() {
    position += stride;
    return *this;
  }
  bool operator!=(const DimensionalIterator &other) const {
    return position != other.position;
  }
  bool operator==(const DimensionalIterator &other) const {
    return position == other.position;
  }
  int64_t getPosition() const { return position; }

private:
  int64_t position, stride;
};

/// Dimensional Range class used to represent the range of
/// a particular dimension of the layout. Can be iterated on
/// using a DimensionalIterator.
class DimensionalRange {
public:
  DimensionalRange() {}
  DimensionalRange(int64_t start, int64_t stop, int64_t step = 1)
      : start(start), stop(stop), step(step) {}
  DimensionalIterator begin() const { return DimensionalIterator(start, step); }
  DimensionalIterator end() const { return DimensionalIterator(stop, step); }

  int64_t start, stop, step;
};

// Iterator class for LayoutAttrs and PerDimLayoutAttrs.
// Provides O(1) access to state for any given dimension.
// Also preserves insertion order.
// Layout iterators skip lane dimensions as these are not
// required during distribution.
class LayoutIterator {
public:
  struct State {
    SmallVector<int64_t>
    computeSIMTIndex(ArrayRef<LayoutDimension> labels) const;
    SmallVector<int64_t>
    computeIteratorProjectedSIMTIndex(ArrayRef<LayoutDimension> labels) const;
    bool contains(LayoutDimension dim) const { return iterators.contains(dim); }
    void erase(LayoutDimension dim) { iterators.erase(dim); }
    DimensionalIterator lookup(LayoutDimension dim) const {
      return iterators.lookup(dim);
    }
    DimensionalIterator &operator[](LayoutDimension dim) {
      return iterators[dim];
    }
    void print() const {
      for (const auto &[dim, it] : iterators) {
        llvm::outs() << stringifyLayoutDimension(dim).str() + ":" +
                            std::to_string(*it) + ", ";
      }
      llvm::outs() << "\n";
    }
    State getProjectedState(int64_t simdIndex) const;
    llvm::MapVector<LayoutDimension, DimensionalIterator> iterators;
    DenseMap<int64_t, DenseSet<LayoutDimension>> simdToLayoutDim;
    llvm::MapVector<LayoutDimension, DimensionalRange> ranges;
  };
  void maybeFreezeAndConcatenate(const LayoutIterator::State &frozenState);
  LayoutIterator(LayoutAttr &attr);
  LayoutIterator(LayoutAttr &attr, int simtIndex);
  LayoutIterator(LayoutAttr &attr, DenseMap<LayoutDimension, int64_t> strides);
  LayoutIterator(LayoutAttr &attr, DenseMap<LayoutDimension, int64_t> strides,
                 int simtIndex);
  LayoutIterator(PerDimLayoutAttr &attr,
                 DenseMap<LayoutDimension, int64_t> strides);
  void apply(std::function<void(const LayoutIterator::State &)>);
  LayoutIterator &operator++();
  State getState() const { return state; }
  LayoutIterator getBatchIterator() const;
  void erase(LayoutDimension dim);
  bool iterationComplete();

private:
  void initialize(const PerDimLayoutAttr &attr,
                  DenseMap<LayoutDimension, int64_t> strides,
                  std::optional<int64_t> dim);
  State state;
  DenseSet<LayoutDimension> frozenDimensions;
  int64_t iterations{0};
  int64_t maxIterations{1};
};

inline bool isLane(LayoutDimension dim) {
  return (dim == LayoutDimension::LANEX) || (dim == LayoutDimension::LANEY) ||
         (dim == LayoutDimension::LANEZ);
}

inline bool isVector(LayoutDimension dim) {
  return (dim == LayoutDimension::VECTORX) ||
         (dim == LayoutDimension::VECTORY) || (dim == LayoutDimension::VECTORZ);
}

inline bool isBatch(LayoutDimension dim) {
  return (dim == LayoutDimension::BATCHX) || (dim == LayoutDimension::BATCHY);
}

AffineExpr computeSIMDIndex(const LayoutIterator::State &state,
                            const PerDimLayoutAttr &attr);

} // namespace mlir::iree_compiler::IREE::VectorExt

#endif // IREE_DIALECTS_DIALECT_VECTOREXT_IR_VECTOREXTOPS_H_
