## copyright 2022 the iree authors
#
# licensed under the apache license v2.0 with llvm exceptions.
# see https://llvm.org/license.txt for license information.
# spdx-license-identifier: apache-2.0 with llvm-exception
"""Defines the collections of device specs and provides query methods."""

from typing import List, Sequence, Set
from e2e_test_framework.definitions import common_definitions
from e2e_test_framework.device_specs import gcp_specs, pixel_4_specs, pixel_6_pro_specs, moto_edge_x30_specs


class DeviceCollections(object):
  """Class to collect and query device specs."""

  def __init__(self, device_specs: Sequence[common_definitions.DeviceSpec]):
    self.device_specs = device_specs

  def query_device_specs(
      self,
      architecture: common_definitions.DeviceArchitecture,
      platform: common_definitions.DevicePlatform,
      device_parameters: Set[str] = set()
  ) -> List[common_definitions.DeviceSpec]:
    matched_device_specs = []
    for device_spec in self.device_specs:
      if device_spec.architecture != architecture:
        continue
      if device_spec.platform != platform:
        continue
      if not device_parameters.issubset(device_spec.device_parameters):
        continue
      matched_device_specs.append(device_spec)

    return matched_device_specs


ALL_DEVICE_SPECS = [
    # Pixel 4
    pixel_4_specs.LITTLE_CORES,
    pixel_4_specs.BIG_CORES,
    pixel_4_specs.GPU,
    # Pixel 6 Pro
    pixel_6_pro_specs.LITTLE_CORES,
    pixel_6_pro_specs.BIG_CORES,
    pixel_6_pro_specs.GPU,
    # Moto Edge X30
    moto_edge_x30_specs.LITTLE_CORES,
    moto_edge_x30_specs.BIG_CORES,
    moto_edge_x30_specs.GPU,
    # GCP machines
    gcp_specs.GCP_C2_STANDARD_16,
    gcp_specs.GCP_A2_HIGHGPU_1G,
]
DEFAULT_DEVICE_COLLECTION = DeviceCollections(ALL_DEVICE_SPECS)
