## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines IREE ARMv8-A benchmarks."""

from typing import List, Tuple

from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework.device_specs import device_collections
from e2e_test_framework.models import tflite_models
import benchmarks.iree.utils
import benchmarks.iree.vmfb_execution_configs


class Android_ARMv8_A_Benchmarks(object):
  """Benchmarks on ARMv8-A Android devices."""
  NONQUANT_MODELS = [
      tflite_models.DEEPLABV3_FP32,
      tflite_models.MOBILESSD_FP32,
      tflite_models.POSENET_FP32,
      tflite_models.MOBILEBERT_FP32,
      tflite_models.MOBILENET_V2,
      tflite_models.MOBILENET_V3SMALL,
  ]
  QUANT_MODELS = [tflite_models.MOBILEBERT_INT8]

  ARMV8_A_CPU_TARGET = iree_definitions.CompileTarget(
      target_architecture=common_definitions.DeviceArchitecture.
      ARMV8_2_A_GENERIC,
      target_backend=iree_definitions.TargetBackend.LLVM_CPU,
      target_abi=iree_definitions.TargetABI.LINUX_ANDROID29)

  DEFAULT_COMPILE_CONFIG = iree_definitions.CompileConfig(
      id=unique_ids.IREE_COMPILE_CONFIG_ANDROID_ARMV8_2_A_GENERIC_DEFAULTS,
      tags=["default-flags"],
      compile_targets=[ARMV8_A_CPU_TARGET])
  MMT4D_COMPILE_CONFIG = iree_definitions.CompileConfig(
      id=unique_ids.IREE_COMPILE_CONFIG_ANDROID_ARMV8_2_A_GENERIC_MMT4D,
      tags=["experimental-flags", "mmt4d"],
      compile_targets=[ARMV8_A_CPU_TARGET],
      extra_flags=["--iree-flow-mmt4d-target-options=arch=aarch64"])
  MMT4D_AND_DOTPROD_COMPILE_CONFIG = iree_definitions.CompileConfig(
      id=unique_ids.IREE_COMPILE_CONFIG_ANDROID_ARMV8_2_A_GENERIC_MMT4D_DOTPROD,
      tags=["experimental-flags", "mmt4d", "dotprod"],
      compile_targets=[ARMV8_A_CPU_TARGET],
      extra_flags=[
          "--iree-flow-mmt4d-target-options=arch=aarch64 features=+dotprod",
          "--iree-llvm-target-cpu-features=+dotprod"
      ])

  def generate(
      self
  ) -> Tuple[List[iree_definitions.ModelCompileConfig],
             List[iree_definitions.E2EModelRunConfig]]:
    """Generates IREE compile and run configs."""

    local_sync_execution_configs = [
        benchmarks.iree.vmfb_execution_configs.ELF_LOCAL_SYNC_CONFIG
    ]
    local_task_execution_configs = [
        benchmarks.iree.vmfb_execution_configs.get_elf_local_task_config(
            thread_num) for thread_num in [1, 4]
    ]

    default_gen_confings = [
        iree_definitions.ModelCompileConfig(
            compile_config=self.DEFAULT_COMPILE_CONFIG, model=model)
        for model in self.NONQUANT_MODELS + self.QUANT_MODELS
    ]
    experimental_gen_confings = [
        iree_definitions.ModelCompileConfig(
            compile_config=self.MMT4D_COMPILE_CONFIG, model=model)
        for model in self.NONQUANT_MODELS
    ] + [
        iree_definitions.ModelCompileConfig(
            compile_config=self.MMT4D_AND_DOTPROD_COMPILE_CONFIG, model=model)
        for model in self.QUANT_MODELS
    ]

    all_devices = device_collections.DEFAULT_DEVICE_COLLECTION.query_device_specs(
        architecture=common_definitions.DeviceArchitecture.ARMV8_2_A_GENERIC,
        platform=common_definitions.DevicePlatform.GENERIC_ANDROID)
    big_cores_devices = device_collections.DEFAULT_DEVICE_COLLECTION.query_device_specs(
        architecture=common_definitions.DeviceArchitecture.ARMV8_2_A_GENERIC,
        platform=common_definitions.DevicePlatform.GENERIC_ANDROID,
        device_parameters={"big-cores"})
    run_configs = benchmarks.iree.utils.generate_e2e_model_run_configs(
        model_compile_configs=default_gen_confings,
        vmfb_execution_configs=local_sync_execution_configs +
        local_task_execution_configs,
        device_specs=all_devices)
    run_configs += benchmarks.iree.utils.generate_e2e_model_run_configs(
        model_compile_configs=experimental_gen_confings,
        vmfb_execution_configs=local_sync_execution_configs,
        device_specs=all_devices)
    run_configs += benchmarks.iree.utils.generate_e2e_model_run_configs(
        model_compile_configs=experimental_gen_confings,
        vmfb_execution_configs=local_task_execution_configs,
        device_specs=big_cores_devices)

    gen_confings = (default_gen_confings + experimental_gen_confings)
    return (gen_confings, run_configs)
