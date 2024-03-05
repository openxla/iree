// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"

#include "llvm/Support/Debug.h"

namespace mlir::iree_compiler {

using IREE::LinalgExt::EncodingAttr;
using IREE::LinalgExt::EncodingRole;

std::string str(SmallVector<int64_t> v) {
  std::stringstream ss;
  ss << "[ " << v[0];
  for (int i = 1; i < v.size(); ++i) {
    ss << " " << v[i];
  }
  ss << " ]";
  return ss.str();
}

static SmallVector<int64_t> reversed(ArrayRef<int64_t> v) {
  SmallVector<int64_t> result(v.size());
  for (unsigned i = 0; i < v.size(); ++i)
    result[v.size() - 1 - i] = v[i];
  return result;
}

static RankedTensorType getTransposedType(RankedTensorType tensorType) {
  SmallVector<int64_t> transposedShape = reversed(tensorType.getShape());
  IREE::LinalgExt::EncodingAttr encoding =
      tensorType.getEncoding()
          .dyn_cast_or_null<IREE::LinalgExt::EncodingAttr>();
  if (encoding) {
    auto transposedRole = encoding.getRole().getValue();
    if (encoding.getRole().getValue() == IREE::LinalgExt::EncodingRole::LHS) {
      transposedRole = IREE::LinalgExt::EncodingRole::RHS;
    }
    if (encoding.getRole().getValue() == IREE::LinalgExt::EncodingRole::RHS) {
      transposedRole = IREE::LinalgExt::EncodingRole::LHS;
    }
    TypeAttr originalTypeAttr = encoding.getOriginalType();
    TypeAttr transposedOriginalTypeAttr;
    if (originalTypeAttr) {
      RankedTensorType originalType = getTransposedType(
          originalTypeAttr.getValue().cast<RankedTensorType>());
      transposedOriginalTypeAttr = TypeAttr::get(originalType);
    }
    auto transposedEncoding = IREE::LinalgExt::EncodingAttr::get(
        encoding.getContext(),
        IREE::LinalgExt::EncodingRoleAttr::get(encoding.getContext(),
                                               transposedRole),
        encoding.getElementTypes(), transposedOriginalTypeAttr,
        encoding.getMatmulNarrow_N(), encoding.getMatmulNarrow_M(),
        encoding.getUserIndexingMaps());
    return RankedTensorType::get(transposedShape, tensorType.getElementType(),
                                 transposedEncoding);
  } else {
    return RankedTensorType::get(transposedShape, tensorType.getElementType());
  }
}

/// For a given tensor type with an encoding, return the materialized
/// type to use for it. If no encoding is set, then return the tensor type
/// itself.
static RankedTensorType
getMaterializedType(RankedTensorType tensorType,
                    MaterializeEncodingFn materializeEncodingFn) {
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(tensorType);
  if (failed(materializeEncodingInfo)) {
    return dropEncoding(tensorType);
  }
  llvm::dbgs() << "\n\n\n--------------------------------------------------"
                  "\nTYPE CONVERTER:\n\n\nMaterializeEncodingInfo:\n"
               << "> innerTileSizes"
               << str(materializeEncodingInfo->innerTileSizes) << "\n"
               << "> innerDimsPos" << str(materializeEncodingInfo->innerDimsPos)
               << "\n"
               << "> outerDimsPerm"
               << str(materializeEncodingInfo->outerDimsPerm) << "\n\n\n"
               << "FROM:\n\n"
               << tensorType << "\n\n";
  IREE::LinalgExt::EncodingAttr encoding =
      tensorType.getEncoding()
          .dyn_cast_or_null<IREE::LinalgExt::EncodingAttr>();
  if (encoding) {
    int64_t matmulNarrowM = getIntOrZero(encoding.getMatmulNarrow_M());
    int64_t matmulNarrowN = getIntOrZero(encoding.getMatmulNarrow_N());
    bool transpose =
        matmulNarrowN && (!matmulNarrowM || matmulNarrowM > matmulNarrowN) &&
        encoding.getRole().getValue() == IREE::LinalgExt::EncodingRole::RESULT;
    if (transpose) {
      llvm::dbgs() << " ...!!!!!! TRANSPOSE !!!!!!...\n";
      return getMaterializedType(getTransposedType(tensorType),
                                 materializeEncodingFn);
    }
  }

  auto resultType =
      tensor::PackOp::inferPackedType(getOriginalTypeWithEncoding(tensorType)
                                          .clone(tensorType.getElementType()),
                                      materializeEncodingInfo->innerTileSizes,
                                      materializeEncodingInfo->innerDimsPos,
                                      materializeEncodingInfo->outerDimsPerm)
          .cast<RankedTensorType>();
  llvm::dbgs() << "TO:\n\n"
               << resultType
               << "\n\n---------------------------------------\n\n";
  return resultType;
}

MaterializeEncodingTypeConverter::MaterializeEncodingTypeConverter(
    MaterializeEncodingFn materializeEncodingFn)
    : materializeEncodingFn(materializeEncodingFn) {
  addConversion([](IntegerType intType) { return intType; });
  addConversion([](IndexType indexType) { return indexType; });
  addConversion([](FloatType floatType) { return floatType; });
  addConversion([](MemRefType memrefType) { return memrefType; });
  addConversion(
      [materializeEncodingFn](RankedTensorType t) -> RankedTensorType {
        return getMaterializedType(t, materializeEncodingFn);
      });
}

MaterializeEncodingConversionTarget::MaterializeEncodingConversionTarget(
    MLIRContext &context)
    : ConversionTarget(context) {
  // Mark any operation that has operands/results with encoding as
  // illegal.
  markUnknownOpDynamicallyLegal([](Operation *op) {
    auto typeHasEncoding = [](Type t) -> bool {
      auto tensorType = t.dyn_cast<RankedTensorType>();
      return tensorType && tensorType.getEncoding();
    };
    auto valueHasEncoding = [=](Value v) -> bool {
      return typeHasEncoding(v.getType());
    };
    bool hasOperandOrResultsWithEncoding =
        llvm::any_of(op->getOperands(), valueHasEncoding) ||
        llvm::any_of(op->getResultTypes(), typeHasEncoding);
    return !hasOperandOrResultsWithEncoding;
  });
}

EncodingAttr getEncodingAttr(RankedTensorType type) {
  return type.getEncoding().dyn_cast_or_null<EncodingAttr>();
}

static AffineMap getMapForRole(EncodingAttr encoding) {
  EncodingRole role = encoding.getRole().getValue();
  if (role == EncodingRole::LHS)
    return cast<AffineMapAttr>(encoding.getUserIndexingMaps()[0])
        .getAffineMap();
  else if (role == EncodingRole::RHS)
    return cast<AffineMapAttr>(encoding.getUserIndexingMaps()[1])
        .getAffineMap();
  else
    return cast<AffineMapAttr>(encoding.getUserIndexingMaps()[2])
        .getAffineMap();
}

FailureOr<linalg::ContractionDimensions>
getEncodingContractionDims(EncodingAttr encoding) {
  auto indexingMapsAttr = encoding.getUserIndexingMaps();
  SmallVector<AffineMap> indexingMaps = llvm::map_to_vector(
      indexingMapsAttr.getValue(), [](Attribute m) -> AffineMap {
        return cast<AffineMapAttr>(m).getAffineMap();
      });
  return linalg::inferContractionDims(indexingMaps);
}

/// Given the dim position of the encoding `user_indexing_maps`, return the
/// matching index of the given encoding's tensor
static unsigned mapDimToRoleIndex(int64_t dimPos, EncodingAttr encoding) {
  AffineMap map = getMapForRole(encoding);
  auto idx = map.getResultPosition(getAffineDimExpr(dimPos, map.getContext()));
  assert(idx.has_value());
  return idx.value();
}

std::optional<SmallVector<int64_t>>
getPermutationToCanonicalMatmulShape(EncodingAttr encoding) {
  FailureOr<linalg::ContractionDimensions> cDims =
      getEncodingContractionDims(encoding);
  if (failed(cDims)) {
    return std::nullopt;
  }
  // Only support at most 1 Batch, M, N, K dimensions for now
  if (cDims->m.size() > 1 || cDims->n.size() > 1 || cDims->k.size() > 1 ||
      cDims->batch.size() > 1) {
    return std::nullopt;
  }
  SmallVector<int64_t> perm;
  EncodingRole role = encoding.getRole().getValue();
  // Add batch dim
  if (!cDims->batch.empty()) {
    perm.push_back(mapDimToRoleIndex(cDims->batch[0], encoding));
  }
  // Add M dim
  if (role != EncodingRole::RHS && cDims->m.size() == 1) {
    perm.push_back(mapDimToRoleIndex(cDims->m[0], encoding));
  }
  // Add K dim
  if (role != EncodingRole::RESULT) {
    perm.push_back(mapDimToRoleIndex(cDims->k[0], encoding));
  }
  // Add N dim
  if (role != EncodingRole::LHS && cDims->n.size() == 1) {
    perm.push_back(mapDimToRoleIndex(cDims->n[0], encoding));
  }
  return perm;
}

RankedTensorType getCanonicalMatmulTypeWithEncoding(RankedTensorType type) {
  auto encoding = getEncodingAttr(type);
  if (!encoding) {
    return type;
  }
  auto perm = getPermutationToCanonicalMatmulShape(encoding);
  if (!perm) {
    return type;
  }
  return RankedTensorType::get(applyPermutation(type.getShape(), perm.value()),
                               type.getElementType(), encoding);
}

RankedTensorType getOriginalTypeWithEncoding(RankedTensorType type) {
  auto encoding = getEncodingAttr(type);
  if (!encoding) {
    return type;
  }
  RankedTensorType originalType = type;
  if (auto originalTypeAttr = encoding.getOriginalType()) {
    originalType = originalTypeAttr.getValue().cast<RankedTensorType>();
  }
  return RankedTensorType::get(originalType.getShape(),
                               originalType.getElementType(), encoding);
}

RankedTensorType dropEncoding(RankedTensorType type) {
  return RankedTensorType::get(type.getShape(), type.getElementType());
}

int64_t getIntOrZero(IntegerAttr a) {
  return a == IntegerAttr() ? 0 : a.getInt();
}

MaterializeEncodingInfo getEncodingInfoForMatmul(EncodingAttr encoding,
                                                 int64_t rank,
                                                 TileMxNxK tileMxNxK) {
  EncodingRole role = encoding.getRole().getValue();
  MaterializeEncodingInfo encodingInfo;
  auto cDims = getEncodingContractionDims(encoding);
  // The following expects M, N, K, and Batch sizes of at most 1 for now
  assert(cDims->m.size() <= 1 && cDims->n.size() <= 1 && cDims->k.size() <= 1 &&
         cDims->batch.size() <= 1 &&
         "Expected at most one M, N, K, and Batch dimension");
  if (!cDims->batch.empty()) {
    encodingInfo.outerDimsPerm.push_back(
        mapDimToRoleIndex(cDims->batch[0], encoding));
  }
  if (role != EncodingRole::RHS && !cDims->m.empty()) {
    encodingInfo.outerDimsPerm.push_back(
        mapDimToRoleIndex(cDims->m[0], encoding));
    encodingInfo.innerDimsPos.push_back(
        mapDimToRoleIndex(cDims->m[0], encoding));
    encodingInfo.innerTileSizes.push_back(tileMxNxK.M);
  }
  if (role != EncodingRole::LHS && !cDims->n.empty()) {
    encodingInfo.outerDimsPerm.push_back(
        mapDimToRoleIndex(cDims->n[0], encoding));
    encodingInfo.innerDimsPos.push_back(
        mapDimToRoleIndex(cDims->n[0], encoding));
    encodingInfo.innerTileSizes.push_back(tileMxNxK.N);
  }
  if (role != EncodingRole::RESULT) {
    encodingInfo.outerDimsPerm.push_back(
        mapDimToRoleIndex(cDims->k[0], encoding));
    encodingInfo.innerDimsPos.push_back(
        mapDimToRoleIndex(cDims->k[0], encoding));
    encodingInfo.innerTileSizes.push_back(tileMxNxK.K);
  }
  return encodingInfo;
}

} // namespace mlir::iree_compiler
