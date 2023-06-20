# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Batch To Space ND tests."""

from absl import app
from iree.tf.support import tf_test_utils
from iree.tf.support import tf_utils
import numpy as np
import tensorflow.compat.v2 as tf


class BatchtoSpaceModule(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([3, 5, 2], tf.float32)])
    def batch_to_space_nd(self, batched):
        block_shape = [3]
        paddings = [[3, 4]]
        return tf.compat.v1.batch_to_space_nd(batched, block_shape, paddings)


class BatchtoSpaceTest(tf_test_utils.TracedModuleTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._modules = tf_test_utils.compile_tf_module(BatchtoSpaceModule)

    def test_space_to_batch_inference(self):
        def space_to_batch_inference(module):
            x = np.linspace(0, 29, 30, dtype=np.float32)
            x = np.reshape(x, [3, 5, 2])
            module.batch_to_space_nd(x)

        self.compare_backends(space_to_batch_inference, self._modules)


def main(argv):
    del argv  # Unused
    if hasattr(tf, "enable_v2_behavior"):
        tf.enable_v2_behavior()
    tf.test.main()


if __name__ == "__main__":
    app.run(main)
