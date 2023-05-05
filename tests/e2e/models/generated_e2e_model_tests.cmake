################################################################################
# Autogenerated by build_tools/testing/generate_cmake_e2e_model_tests.py       #
# To update the tests, modify definitions in the generator and regenerate this #
# file.                                                                        #
################################################################################

iree_benchmark_suite_module_test(
  NAME
    "mobilenet_v1_fp32_correctness_test"
  DRIVER
    "local-sync"
  EXPECTED_OUTPUT
    "mobilenet_v1_fp32_expected_output.txt"
  MODULES
    "riscv_64-Linux=iree_MobileNetV1_fp32_module_e80d71ed8e86c0756226b2323e27e2c7c0fff8eddde59ba69e9222d36ee3eef6/module.vmfb"
    "x86_64-Linux=iree_MobileNetV1_fp32_module_02cebfbec13685725c5b3c805c6c620ea3f885027bfbb14d17425798e391486f/module.vmfb"
  RUNNER_ARGS
    "--function=main"
    "--input=1x224x224x3xf32=0"
    "--device_allocator=caching"
)

iree_benchmark_suite_module_test(
  NAME
    "efficientnet_int8_correctness_test"
  DRIVER
    "local-sync"
  EXPECTED_OUTPUT
    "efficientnet_int8_expected_output.txt"
  MODULES
    "x86_64-Linux=iree_EfficientNet_int8_module_3926415c1504dfc277674fee17bdfbd68090634b8b52620d8d5755082a89a16d/module.vmfb"
  RUNNER_ARGS
    "--function=main"
    "--input=1x224x224x3xui8=0"
    "--device_allocator=caching"
)

iree_benchmark_suite_module_test(
  NAME
    "deeplab_v3_fp32_correctness_test"
  DRIVER
    "local-sync"
  EXPECTED_OUTPUT
    "deeplab_v3_fp32_input_0_expected_output.npy"
  MODULES
    "arm_64-Android=iree_DeepLabV3_fp32_module_f06fff09f8cebc27d1674045712aaa60afe7aef388c4bc505897f55c3a0d8abb/module.vmfb"
    "x86_64-Linux=iree_DeepLabV3_fp32_module_87aead729018ce5f114501cecefb6315086eb2a21ae1b30984b1794f619871c6/module.vmfb"
  RUNNER_ARGS
    "--function=main"
    "--input=1x257x257x3xf32=0"
    "--device_allocator=caching"
    "--expected_f32_threshold=0.001"
)

iree_benchmark_suite_module_test(
  NAME
    "person_detect_int8_correctness_test"
  DRIVER
    "local-sync"
  EXPECTED_OUTPUT
    "1x2xi8=[72 -72]"
  MODULES
    "riscv_32-Linux=iree_PersonDetect_int8_module_1ef2da238443010024d69ceb6fe6ab6fa8cf5f4ce7d424dace3a572592043e70/module.vmfb"
    "riscv_64-Linux=iree_PersonDetect_int8_module_14a15b9072caaee5e2a274a9bbc436a56d095611e5a8e9841f110741d34231f9/module.vmfb"
    "x86_64-Linux=iree_PersonDetect_int8_module_eb56e91246a131fa41bd335c1c072ffb6e7ffe651ecf65f4eeb171b12848b0ed/module.vmfb"
  RUNNER_ARGS
    "--function=main"
    "--input=1x96x96x1xi8=0"
    "--device_allocator=caching"
)

