################################################################################
# Autogenerated by build_tools/bazel_to_cmake/bazel_to_cmake.py from           #
# compiler/src/iree/compiler/Modules/IO/Parameters/Transforms/BUILD.bazel      #
#                                                                              #
# Use iree_cmake_extra_content from iree/build_defs.oss.bzl to add arbitrary   #
# CMake-only content.                                                          #
#                                                                              #
# To disable autogeneration for this file entirely, delete this header.        #
################################################################################

iree_add_all_subdirs()

iree_cc_library(
  NAME
    Transforms
  HDRS
    "ArchiveUtils.h"
    "Passes.h"
    "Passes.h.inc"
  SRCS
    "ArchiveUtils.cpp"
    "ExportParameters.cpp"
    "GenerateSplatParameterArchive.cpp"
    "Passes.cpp"
  DEPS
    ::PassesIncGen
    LLVMSupport
    MLIRArithDialect
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransformUtils
    MLIRTransforms
    iree::base
    iree::compiler::Dialect::Stream::IR
    iree::compiler::Dialect::Util::IR
    iree::hal
    iree::io::formats::irpa
    iree::io::parameter_index
    iree::io::scope_map
    iree::tooling::parameter_util
  PUBLIC
)

iree_tablegen_library(
  NAME
    PassesIncGen
  TD_FILE
    "Passes.td"
  OUTS
    --gen-pass-decls Passes.h.inc
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
