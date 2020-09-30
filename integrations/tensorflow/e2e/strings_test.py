# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import app
import numpy as np
from pyiree.tf.support import tf_test_utils
import string
import tensorflow.compat.v2 as tf


class StringsModule(tf.Module):
  """A Module for converting a set of ids to the concatenated string."""

  def __init__(self):
    wordparts = [str(c) for c in string.printable]
    self.wordparts = tf.constant(wordparts, tf.string)

  @tf.function(input_signature=[
      tf.TensorSpec((None, None), dtype=tf.int32),
  ])
  def print_ids(self, ids):
    string_tensor = tf.strings.as_string(ids)
    tf.print(string_tensor)

  @tf.function(input_signature=[
      tf.TensorSpec((None, None), dtype=tf.int32),
  ])
  def strings_to_ids(self, ids):
    wps = tf.gather(self.wordparts, ids)
    return tf.strings.reduce_join(wps, 1)


class StringsTest(tf_test_utils.TracedModuleTestCase):

  def test_print_ids(self):

    def print_ids(module):
      input_ids = np.asarray(
          [[12, 10, 29, 28, 94, 15, 24, 27, 94, 25, 21, 10, 34],
           [13, 24, 16, 28, 94, 15, 24, 27, 94, 28, 29, 10, 34]])
      module.print_ids(input_ids)

    self.compare_backends(print_ids)

  def test_strings_to_ids(self):

    def strings_to_ids(module):
      input_ids = np.asarray(
          [[12, 10, 29, 28, 94, 15, 24, 27, 94, 25, 21, 10, 34],
           [13, 24, 16, 28, 94, 15, 24, 27, 94, 28, 29, 10, 34]])
      module.strings_to_ids(input_ids)

    self.compare_backends(strings_to_ids)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf_test_utils.compile_tf_module(StringsModule)
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
