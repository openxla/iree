################################################################################
# Autogenerated by build_tools/bazel_to_cmake/bazel_to_cmake.py from           #
# compiler/plugins/input/TOSA/BUILD.bazel                                      #
#                                                                              #
# Use iree_cmake_extra_content from iree/build_defs.oss.bzl to add arbitrary   #
# CMake-only content.                                                          #
#                                                                              #
# To disable autogeneration for this file entirely, delete this header.        #
################################################################################

iree_add_all_subdirs()

iree_compiler_register_plugin(
  PLUGIN_ID
    input_tosa
  TARGET
    ::registration
)

iree_cc_library(
  NAME
    registration
  SRCS
    "PluginRegistration.cpp"
  DEPS
    MLIRIR
    MLIRPass
    MLIRTosaDialect
    MLIRTosaToArith
    MLIRTosaToLinalg
    MLIRTosaToSCF
    MLIRTosaToTensor
    MLIRTransforms
    iree::compiler::PluginAPI
    iree::compiler::plugins::input::TOSA::InputConversion
  PUBLIC
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
