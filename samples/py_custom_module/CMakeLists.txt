# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_BUILD_PYTHON_BINDINGS)
  return()
endif()

# These tests perform linking via the Compiler API, which is only supported
# in bundled-LLVM builds at the moment (#14086).
if(NOT IREE_BUILD_BUNDLED_LLVM)
  return()
endif()

iree_py_test(
  NAME
    decode_secret_message_sample
  SRCS
    "decode_secret_message.py"
)
