// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"

#include <optional>
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::iree_compiler::IREE::GPU {

namespace {

//===----------------------------------------------------------------------===//
// Target details structs
//===----------------------------------------------------------------------===//

// Note that the following structs basically mirror the corresponding attribute
// definitions in IREEGPUAttrs.td. The purpose of having mirroring structs here
// is to allow us building a known GPU target "database" in POD data structures
// without all the MLIR attribute wrappers; so life is easier to put such
// information in static constant data.
//
// Ideally we should be able to generate The following structs from the TableGen
// attribute definitions, but it takes quite some efforts to plumb all pieces
// through. So now fine with some duplication.

// Workgroup processor level feature/limit details
struct WgpDetails {
  ComputeBitwidths compute;
  StorageBitwidths storage;
  SubgroupOps subgroupOps;
  DotProductOps dotproductOps;
  uint32_t mmaCount;
  const MMAIntrinsic *mmaOps;

  std::array<int32_t, 2> subgroupSizeChoices;
  std::array<int32_t, 3> maxWorkgroupSizes;
  uint32_t maxThreadSize;
  uint32_t maxWorkgroupMemoryBytes;
};

// Chip level feature/limit details
struct ChipDetails {
  uint32_t wgpCount;
};

// Full target details
struct TargetDetails {
  const WgpDetails *wgp;
  const ChipDetails *chip;
};

//===----------------------------------------------------------------------===//
// Utility definitions
//===----------------------------------------------------------------------===//

const ComputeBitwidths allFPComputeBits =
    ComputeBitwidths::FP64 | ComputeBitwidths::FP32 | ComputeBitwidths::FP16;
const ComputeBitwidths allIntComputeBits =
    ComputeBitwidths::Int64 | ComputeBitwidths::Int32 |
    ComputeBitwidths::Int16 | ComputeBitwidths::Int8;
const ComputeBitwidths allComputeBits = allFPComputeBits | allIntComputeBits;

const StorageBitwidths allStorageBits =
    StorageBitwidths::B8 | StorageBitwidths::B16 | StorageBitwidths::B32 |
    StorageBitwidths::B64;

const SubgroupOps allSubgroupOps =
    SubgroupOps::Shuffle | SubgroupOps::Arithmetic;

const DotProductOps allDotProductOps = DotProductOps::DP4xI8ToI32;

#define ARRAY_SIZE(array) (sizeof(array) / sizeof(array[0]))

// Creates the corresponding TargetAttr from the given target |details|.
TargetAttr createTargetAttr(const TargetDetails &details, StringRef arch,
                            StringRef features, MLIRContext *context) {
  const WgpDetails *wgp = details.wgp;

  SmallVector<MMAAttr, 8> mmaAttrs;
  mmaAttrs.reserve(wgp->mmaCount);
  for (int i = 0; i < wgp->mmaCount; ++i)
    mmaAttrs.push_back(MMAAttr::get(context, wgp->mmaOps[i]));

  SmallVector<int32_t, 2> subgroupSizes;
  subgroupSizes.push_back(wgp->subgroupSizeChoices.front());
  if (wgp->subgroupSizeChoices.back() != wgp->subgroupSizeChoices.front())
    subgroupSizes.push_back(wgp->subgroupSizeChoices.back());

  auto targetWgp = TargetWgpAttr::get(
      context, ComputeBitwidthsAttr::get(context, details.wgp->compute),
      StorageBitwidthsAttr::get(context, wgp->storage),
      SubgroupOpsAttr::get(context, wgp->subgroupOps),
      DotProductOpsAttr::get(context, wgp->dotproductOps),
      MMAOpsArrayAttr::get(context, mmaAttrs),
      DenseI32ArrayAttr::get(context, subgroupSizes),
      DenseI32ArrayAttr::get(context, wgp->maxWorkgroupSizes),
      wgp->maxThreadSize, wgp->maxWorkgroupMemoryBytes, DictionaryAttr{});

  TargetChipAttr targetChip;
  if (details.chip)
    targetChip =
        TargetChipAttr::get(context, details.chip->wgpCount, DictionaryAttr{});

  return TargetAttr::get(context, arch, features, targetWgp, targetChip);
}

//===----------------------------------------------------------------------===//
// Known AMD target details
//===----------------------------------------------------------------------===//

const WgpDetails *getCDNA3WgpDetails() {
  static const MMAIntrinsic cdna3MMAOps[] = {
      MMAIntrinsic::MFMA_F16_16x16x16_F32,
      MMAIntrinsic::MFMA_F16_32x32x8_F32,
  };
  static const WgpDetails cdna3Wgp = {
      allComputeBits,   allStorageBits,          allSubgroupOps,
      allDotProductOps, ARRAY_SIZE(cdna3MMAOps), cdna3MMAOps,
      {64, 64},         {1024, 1024, 1024},      1024,
      64 * 1024};
  return &cdna3Wgp;
}

const WgpDetails *getCDNA2WgpDetails() {
  static const MMAIntrinsic cdna2MMAOps[] = {
      MMAIntrinsic::MFMA_F16_16x16x16_F32,
      MMAIntrinsic::MFMA_F16_32x32x8_F32,
  };
  static const WgpDetails cdna2Wgp = {
      allComputeBits,   allStorageBits,          allSubgroupOps,
      allDotProductOps, ARRAY_SIZE(cdna2MMAOps), cdna2MMAOps,
      {64, 64},         {1024, 1024, 1024},      1024,
      64 * 1024};
  return &cdna2Wgp;
}

const WgpDetails *getCDNA1WgpDetails() {
  static const MMAIntrinsic cdna1MMAOps[] = {
      MMAIntrinsic::MFMA_F16_16x16x16_F32,
      MMAIntrinsic::MFMA_F16_32x32x8_F32,
  };
  static const WgpDetails cdna1Wgp = {
      allComputeBits,   allStorageBits,          allSubgroupOps,
      allDotProductOps, ARRAY_SIZE(cdna1MMAOps), cdna1MMAOps,
      {64, 64},         {1024, 1024, 1024},      1024,
      64 * 1024};
  return &cdna1Wgp;
}

const WgpDetails *getRDNA3WgpDetails() {
  static const MMAIntrinsic rdna3MMAOps[] = {
      MMAIntrinsic::WMMA_F16_16x16x16_F32,
  };
  static const WgpDetails rdna3Wgp = {
      allComputeBits,   allStorageBits,          allSubgroupOps,
      allDotProductOps, ARRAY_SIZE(rdna3MMAOps), rdna3MMAOps,
      {32, 64},         {1024, 1024, 1024},      1024,
      64 * 1024};
  return &rdna3Wgp;
}

const WgpDetails *getRDNA2WgpDetails() {
  static const WgpDetails rdna2Wgp = {
      allComputeBits, allStorageBits,     allSubgroupOps, allDotProductOps,
      /*mmaCount=*/0, /*mmaOps=*/nullptr, {32, 64},       {1024, 1024, 1024},
      1024,           64 * 1024};
  return &rdna2Wgp;
}

const WgpDetails *getRDNA1WgpDetails() {
  static const WgpDetails rdna1Wgp = {
      allComputeBits, allStorageBits,     allSubgroupOps, DotProductOps::None,
      /*mmaCount=*/0, /*mmaOps=*/nullptr, {32, 64},       {1024, 1024, 1024},
      1024,           64 * 1024};
  return &rdna1Wgp;
}

std::optional<TargetDetails> getAMDGPUTargetDetails(StringRef target) {
  const WgpDetails *cdna3Wgp = getCDNA3WgpDetails();
  const WgpDetails *cdna2Wgp = getCDNA2WgpDetails();
  const WgpDetails *cdna1Wgp = getCDNA1WgpDetails();
  const WgpDetails *rdna3Wgp = getRDNA3WgpDetails();
  const WgpDetails *rdna2Wgp = getRDNA2WgpDetails();
  const WgpDetails *rdna1Wgp = getRDNA1WgpDetails();

  // "AMD Instinct MI300 Series Product Offerings" in Page 23 of
  // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
  static const ChipDetails mi300xChip = {304};
  static const ChipDetails mi300aChip = {228};

  // "AMD Instinct MI200 Series Accelerator Product Offerings" in Page 14 of
  // https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf
  static const ChipDetails mi250xChip = {220};
  static const ChipDetails mi250Chip = {208};
  static const ChipDetails mi210Chip = {104};

  // "AMD CDNA Architecture Compute Units" in Page 5 of
  // https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna-white-paper.pdf
  static const ChipDetails mi100Chip = {120};

  static const ChipDetails rx7900xtxChip = {96};
  static const ChipDetails rx7900xtChip = {84};
  static const ChipDetails rx7800xtChip = {60};
  static const ChipDetails rx7700xtChip = {54};

  // See https://llvm.org/docs/AMDGPUUsage.html#processors for gfxN to
  // cdnaN/rdnaN mapping.

  return llvm::StringSwitch<std::optional<TargetDetails>>(target.lower())
      .Case("mi300x", TargetDetails{cdna3Wgp, &mi300xChip})
      .Case("mi300a", TargetDetails{cdna3Wgp, &mi300aChip})
      .Cases("cdna3", "gfx940", "gfx941", "gfx942",
             TargetDetails{cdna3Wgp, nullptr})
      .Case("mi250x", TargetDetails{cdna2Wgp, &mi250xChip})
      .Case("mi250", TargetDetails{cdna2Wgp, &mi250Chip})
      .Case("mi210", TargetDetails{cdna2Wgp, &mi210Chip})
      .Cases("cdna2", "gfx90a", TargetDetails{cdna2Wgp, nullptr})
      .Case("mi100", TargetDetails{cdna1Wgp, &mi100Chip})
      .Cases("cdna1", "gfx908", TargetDetails{cdna1Wgp, nullptr})
      // https://www.techpowerup.com/gpu-specs/radeon-rx-7900-xtx.c3941
      .Case("rx7900xtx", TargetDetails{rdna3Wgp, &rx7900xtxChip})
      // https://www.techpowerup.com/gpu-specs/radeon-rx-7900-xt.c3912
      .Case("rx7900xt", TargetDetails{rdna3Wgp, &rx7900xtChip})
      // https://www.techpowerup.com/gpu-specs/radeon-rx-7800-xt.c3839
      .Case("rx7800xt", TargetDetails{rdna3Wgp, &rx7800xtChip})
      // https://www.techpowerup.com/gpu-specs/radeon-rx-7700-xt.c3911
      .Case("rx7700xt", TargetDetails{rdna3Wgp, &rx7700xtChip})
      .Cases("rdna3", "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150",
             "gfx1151", TargetDetails{rdna3Wgp, nullptr})
      .Cases("rdna2", "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034",
             "gfx1035", "gfx1036", TargetDetails{rdna2Wgp, nullptr})
      .Cases("rdna1", "gfx1010", "gfx1011", "gfx1012", "gfx1013",
             TargetDetails{rdna1Wgp, nullptr})
      .Default(std::nullopt);
}

StringRef normalizeAMDGPUTarget(StringRef target) {
  if (target.starts_with("gfx"))
    return target;

  return llvm::StringSwitch<StringRef>(target.lower())
      .Case("mi300x", "gfx942")
      .Case("mi300a", "gfx940")
      .Cases("mi250x", "mi250", "mi210", "cdna2", "gfx90a")
      .Cases("rx7900xtx", "rx7900xt", "gfx1100")
      .Cases("rx7800xt", "rx7700xt", "gfx1101")
      .Default(StringRef());
}

//===----------------------------------------------------------------------===//
// Known ARM target details
//===----------------------------------------------------------------------===//

const WgpDetails *getValhallWgpDetails() {
  ComputeBitwidths computeBitwdiths =
      allIntComputeBits | ComputeBitwidths::FP32 | ComputeBitwidths::FP16;
  // clang-format off
  static const WgpDetails valhallWgp = {
      computeBitwdiths,   allStorageBits,     allSubgroupOps,  allDotProductOps,
      /*mmaCount=*/0,     /*mmaOps=*/nullptr, {16},            {512, 512, 512},
      512,                32 * 1024};
  // clang-format on
  return &valhallWgp;
}

std::optional<TargetDetails> getARMGPUTargetDetails(StringRef target) {
  const WgpDetails *valhallWgp = getValhallWgpDetails();

  // Note that the underlying GPU may have certain capabilities but the Android
  // version and driver stack may not expose them. So the following is just and
  // will always be approximate.

  return llvm::StringSwitch<std::optional<TargetDetails>>(target.lower())
      // Mali-G715: https://vulkan.gpuinfo.org/displayreport.php?id=29754
      .Cases("mali-g715", "mali-g615", "valhall4",
             TargetDetails{valhallWgp, nullptr})
      // Mali-G710: https://vulkan.gpuinfo.org/displayreport.php?id=30471
      .Cases("mali-g710", "mali-g510", "mali-g310", "valhall3",
             TargetDetails{valhallWgp, nullptr})
      // Mali-G78: https://vulkan.gpuinfo.org/displayreport.php?id=29994
      .Cases("mali-g78", "valhall2", TargetDetails{valhallWgp, nullptr})
      // Mali-G57: https://vulkan.gpuinfo.org/displayreport.php?id=24636
      .Cases("mali-g77", "mali-g57", "valhall1", "valhall",
             TargetDetails{valhallWgp, nullptr})
      .Default(std::nullopt);
}

StringRef normalizeARMGPUTarget(StringRef target) {
  if (target == "valhall")
    return "valhall1";
  if (target.starts_with("valhall"))
    return target;

  return llvm::StringSwitch<StringRef>(target.lower())
      .Cases("mali-g715", "mali-g615", "valhall4")
      .Cases("mali-g710", "mali-g510", "mali-g310", "valhall3")
      .Case("mali-78", "valhall2")
      .Cases("mali-g77", "mali-g57", "valhall1")
      .Default(StringRef());
}

//===----------------------------------------------------------------------===//
// Known NVIDIA target details
//===----------------------------------------------------------------------===//

// FIXME: In the following query functions, we are using AMD WMMA intrinsics
// that have different layout from NVIDIA WMMA intrinsics. This is fine given
// right now we only use this to indicate target features for Vulkan, where all
// cooperative matrix layouts are opaque. We need to create NVIDIA specific WMMA
// intrinsics if we need to have explicit layout analysis and register mapping.

const WgpDetails *getAmpereWgpDetails() {
  static const MMAIntrinsic mmaOps[] = {
      MMAIntrinsic::WMMA_F16_16x16x16_F32,
      MMAIntrinsic::WMMA_F16_16x16x16_F16,
  };
  static const WgpDetails ampereWgp = {
      allComputeBits,   allStorageBits,     allSubgroupOps,
      allDotProductOps, ARRAY_SIZE(mmaOps), mmaOps,
      {32, 32},         {1024, 1024, 1024}, 1024,
      163 * 1024};
  return &ampereWgp;
}

const WgpDetails *getTuringWgpDetails() {
  static const MMAIntrinsic mmaOps[] = {
      MMAIntrinsic::WMMA_F16_16x16x16_F32,
      MMAIntrinsic::WMMA_F16_16x16x16_F16,
  };
  static const WgpDetails turingWgp = {
      allComputeBits,   allStorageBits,     allSubgroupOps,
      allDotProductOps, ARRAY_SIZE(mmaOps), mmaOps,
      {32, 32},         {1024, 1024, 1024}, 1024,
      64 * 1024};
  return &turingWgp;
}

const WgpDetails *getVoltaWgpDetails() {
  static const MMAIntrinsic mmaOps[] = {
      MMAIntrinsic::WMMA_F16_16x16x16_F32,
      MMAIntrinsic::WMMA_F16_16x16x16_F16,
  };
  // clang-format off
  static const WgpDetails voltaWgp = {
      allComputeBits,     allStorageBits, allSubgroupOps, DotProductOps::None,
      ARRAY_SIZE(mmaOps), mmaOps,         {32, 32},       {1024, 1024, 1024},
      1024,               96 * 1024};
  // clang-format on
  return &voltaWgp;
}

const WgpDetails *getPascalWgpDetails() {
  // clang-format off
  static const WgpDetails pascalWgp = {
      allComputeBits, allStorageBits, allSubgroupOps, DotProductOps::None,
      0, nullptr, // Pascal does not have tensor core support.
      {32, 32}, {1024, 1024, 1024}, 1024, 48 * 1024};
  // clang-format on
  return &pascalWgp;
}

std::optional<TargetDetails> getNVIDIAGPUTargetDetails(StringRef target) {
  const WgpDetails *ampereWgp = getAmpereWgpDetails();
  const WgpDetails *turingWgp = getTuringWgpDetails();
  const WgpDetails *voltaWgp = getVoltaWgpDetails();
  const WgpDetails *pascalWgp = getPascalWgpDetails();

  static const ChipDetails a100Chip = {108};
  static const ChipDetails rtx3090tiChip = {84};
  static const ChipDetails rtx3090Chip = {82};
  static const ChipDetails rtx3080tiChip = {80};
  static const ChipDetails rtx3080Chip = {68};
  static const ChipDetails rtx3070tiChip = {48};
  static const ChipDetails rtx3070Chip = {46};

  // https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
  // lists mappings from microarchitectures to compute capabilities.

  return llvm::StringSwitch<std::optional<TargetDetails>>(target.lower())
      // https://www.techpowerup.com/gpu-specs/a100-sxm4-80-gb.c3746
      .Case("a100", TargetDetails{ampereWgp, &a100Chip})
      // https://www.techpowerup.com/gpu-specs/geforce-rtx-3090-ti.c3829
      .Case("rtx3090ti", TargetDetails{ampereWgp, &rtx3090tiChip})
      // https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622
      .Case("rtx3090", TargetDetails{ampereWgp, &rtx3090Chip})
      // https://www.techpowerup.com/gpu-specs/geforce-rtx-3080-ti.c3735
      .Case("rtx3080ti", TargetDetails{ampereWgp, &rtx3080tiChip})
      // https://www.techpowerup.com/gpu-specs/geforce-rtx-3080.c3621
      .Case("rtx3080", TargetDetails{ampereWgp, &rtx3080Chip})
      // https://www.techpowerup.com/gpu-specs/geforce-rtx-3070-ti.c3675
      .Case("rtx3070ti", TargetDetails{ampereWgp, &rtx3070tiChip})
      // https://www.techpowerup.com/gpu-specs/geforce-rtx-3070.c3674
      .Case("rtx3070", TargetDetails{ampereWgp, &rtx3070Chip})
      .Cases("ampere", "sm_80", "sm_86", "sm_87",
             TargetDetails{ampereWgp, nullptr})
      .Cases("turing", "sm_75", TargetDetails{turingWgp, nullptr})
      .Cases("volta", "sm_70", "sm_72", TargetDetails{voltaWgp, nullptr})
      .Cases("pascal", "sm_60", "sm_61", "sm_62",
             TargetDetails{pascalWgp, nullptr})
      .Default(std::nullopt);
}

StringRef normalizeNVIDIAGPUTarget(StringRef target) {
  if (target.starts_with("sm_"))
    return target;

  if (target.starts_with("rtx40"))
    return "sm_89";
  if (target.starts_with("rtx30"))
    return "sm_86";
  if (target.starts_with("rtx20"))
    return "sm_75";

  return llvm::StringSwitch<StringRef>(target.lower())
      .Case("a100", "sm_80")
      .Case("ampere", "sm_80") // Or sm_86/87; use smaller version
      .Case("turing", "sm_75")
      .Case("volta", "sm_70")  // Or sm_72; use smaller version
      .Case("pascal", "sm_60") // Or sm_61/62; use smaller version
      .Default(StringRef());
}

//===----------------------------------------------------------------------===//
// Known Qualcomm target details
//===----------------------------------------------------------------------===//

const WgpDetails *getAdrenoWgpDetails() {
  auto computeBitwdiths = ComputeBitwidths::Int32 | ComputeBitwidths::Int16 |
                          ComputeBitwidths::Int8 | ComputeBitwidths::FP32 |
                          ComputeBitwidths::FP16;
  auto storageBitwidths =
      StorageBitwidths::B64 | StorageBitwidths::B32 | StorageBitwidths::B16;
  // clang-format off
  static const WgpDetails adrenoWgp = {
      computeBitwdiths,   storageBitwidths,   allSubgroupOps,
      allDotProductOps,   /*mmaCount=*/0,     /*mmaOps=*/nullptr,
      {64},               {1024, 1024, 1024}, 1024,
      32 * 1024};
  // clang-format on
  return &adrenoWgp;
}

std::optional<TargetDetails> getQualcommGPUTargetDetails(StringRef target) {
  const WgpDetails *adrenoWgp = getAdrenoWgpDetails();

  // Note that the underlying GPU may have certain capabilities but the Android
  // version and driver stack may not expose them. So the following is just and
  // will always be approximate.

  // Adreno GPUs are quite opaque regarding their generational information.

  return llvm::StringSwitch<std::optional<TargetDetails>>(target.lower())
      // Adreno-750: https://vulkan.gpuinfo.org/displayreport.php?id=27414
      // Adreno-740: https://vulkan.gpuinfo.org/displayreport.php?id=19218
      // Adreno-730: https://vulkan.gpuinfo.org/displayreport.php?id=19382
      .Cases("adreno-750", "adreno-740", "adreno-730", "adreno",
             TargetDetails{adrenoWgp, nullptr})
      .Default(std::nullopt);
}

StringRef normalizeQualcommGPUTarget(StringRef target) {
  if (target == "adreno")
    return target;

  StringRef t = target;
  if (!t.consume_front("adreno-"))
    return "";

  // The can exist an optional L at the end.
  if (t.ends_with("l"))
    t = t.drop_back();

  // Check whether we have a product number
  unsigned number = 0;
  // StringRef::consumeInteger() returns true to signify errors.
  if (t.size() != 3 || t.consumeInteger(10, number))
    return "";

  return target;
}

} // namespace

//===----------------------------------------------------------------------===//
// Query functions
//===----------------------------------------------------------------------===//

TargetAttr getHIPTargetDetails(StringRef target, StringRef features,
                               MLIRContext *context) {
  if (auto details = getAMDGPUTargetDetails(target)) {
    return createTargetAttr(*details, normalizeAMDGPUTarget(target), features,
                            context);
  }
  return nullptr;
}

StringRef normalizeHIPTarget(StringRef target) {
  return normalizeAMDGPUTarget(target);
}

TargetAttr getCUDATargetDetails(StringRef target, StringRef features,
                                MLIRContext *context) {
  if (auto details = getNVIDIAGPUTargetDetails(target))
    return createTargetAttr(*details, normalizeNVIDIAGPUTarget(target),
                            features, context);
  return nullptr;
}

StringRef normalizeCUDATarget(StringRef target) {
  return normalizeNVIDIAGPUTarget(target);
}

TargetAttr getVulkanTargetDetails(llvm::StringRef target,
                                  MLIRContext *context) {
  // Go through each vendor's target details. This assumes we won't have
  // duplicated product or microarchitecture names among vendors, which should
  // be the case.

  if (auto details = getAMDGPUTargetDetails(target)) {
    return createTargetAttr(*details, normalizeAMDGPUTarget(target),
                            /*features=*/"", context);
  }
  if (auto details = getARMGPUTargetDetails(target)) {
    return createTargetAttr(*details, normalizeARMGPUTarget(target),
                            /*features=*/"", context);
  }
  if (auto details = getNVIDIAGPUTargetDetails(target)) {
    return createTargetAttr(*details, normalizeNVIDIAGPUTarget(target),
                            /*features=*/"", context);
  }
  if (auto details = getQualcommGPUTargetDetails(target)) {
    return createTargetAttr(*details, normalizeQualcommGPUTarget(target),
                            /*features=*/"", context);
  }
  return nullptr;
}

TargetAttr getFullTarget(StringRef targetAPI, StringRef aliasTarget,
                         StringRef features, MLIRContext *context) {
  return llvm::StringSwitch<TargetAttr>(targetAPI)
      .Case("cuda", getCUDATargetDetails(aliasTarget, features, context))
      .Case("rocm", getHIPTargetDetails(aliasTarget, features, context))
      .Default(nullptr);
}

} // namespace mlir::iree_compiler::IREE::GPU
