## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools
from typing import List, Sequence
from e2e_test_framework.definitions import common_definitions, iree_definitions

MODULE_BENCHMARK_TOOL = "iree-benchmark-module"


def generate_e2e_model_run_configs(
    model_compile_configs: Sequence[iree_definitions.ModelCompileConfig],
    vmfb_execution_configs: Sequence[iree_definitions.VMFBExecutionConfig],
    device_specs: Sequence[common_definitions.DeviceSpec],
) -> List[iree_definitions.E2EModelRunConfig]:
  """Generates the run specs from the product of compile specs and run configs.
  """
  return [
      iree_definitions.E2EModelRunConfig(
          model_compile_config=model_compile_config,
          vmfb_execution_config=vmfb_execution_config,
          target_device_spec=device_spec,
          input_data=common_definitions.RANDOM_MODEL_INPUT_DATA)
      for model_compile_config, vmfb_execution_config, device_spec in itertools.
      product(model_compile_configs, vmfb_execution_configs, device_specs)
  ]
