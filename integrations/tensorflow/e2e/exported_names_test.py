# Lint as: python3
# Copyright 2019 Google LLC
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

from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf


class DontExportEverything(tf.Module):

  @tf.function(input_signature=[])
  def exported_fn(self):
    return tf.constant([42.])

  # No input_signature, so it cannot be imported by the SavedModel importer.
  # We need to ensure that
  @tf.function(input_signature=[])
  def unreachable_fn(self):
    return tf.constant([24.])


# To pass a set of exported names for the module, instead of passing just a
# module ctor, instead pass a pair `(ctor, [list, of, exported, names])`.
@tf_test_utils.compile_module(
    DontExportEverything, exported_names=["exported_fn"])
class DontExportEverythingTest(tf_test_utils.SavedModelTestCase):

  def test_exported_name(self):
    self.get_module().exported_fn().assert_all_close()

  def test_unreachable_name(self):
    with self.assertRaises(AttributeError):
      self.get_module().unreachable_fn()


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
