# coding=utf-8
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for compiler_opt.rl.weighted_bc_trainer."""

import functools
from absl import app
from tf_agents.system import system_multiprocessing as multiprocessing
import tensorflow as tf

from compiler_opt.rl.imitation_learning.generate_bc_trajectories_lib import SequenceExampleFeatureNames
from compiler_opt.rl.imitation_learning.weighted_bc_trainer import TrainingWeights


class TestTrainingWeights(tf.test.TestCase):
  """Tests for TrainingWeights class."""

  def test_bucket_by_feature(self):
    # pylint: disable=protected-access
    train_weights = TrainingWeights(partitions=(0.5, 10.0, 20.0))
    data = [
        {
            SequenceExampleFeatureNames.module_name: 'module_1',
            SequenceExampleFeatureNames.loss: 47.0,
            SequenceExampleFeatureNames.horizon: 10
        },
        {
            SequenceExampleFeatureNames.module_name: 'module_2',
            SequenceExampleFeatureNames.loss: .5,
            SequenceExampleFeatureNames.horizon: 10
        },
        {
            SequenceExampleFeatureNames.module_name: 'module_3',
            SequenceExampleFeatureNames.loss: 19.0,
            SequenceExampleFeatureNames.horizon: 10
        },
    ]
    buckets = train_weights._bucket_by_feature(
        data=data, feature_name=SequenceExampleFeatureNames.loss)
    self.assertEqual(buckets, [[], [data[1]], [data[2]], [data[0]]])

  def test_create_new_profile(self):
    train_weights = TrainingWeights(partitions=(0.5, 10.0, 20.0))
    data_comp = [
        {
            SequenceExampleFeatureNames.module_name: 'module_1',
            SequenceExampleFeatureNames.loss: 47.0,
            SequenceExampleFeatureNames.horizon: 10
        },
        {
            SequenceExampleFeatureNames.module_name: 'module_2',
            SequenceExampleFeatureNames.loss: .0,
            SequenceExampleFeatureNames.horizon: 10
        },
        {
            SequenceExampleFeatureNames.module_name: 'module_3',
            SequenceExampleFeatureNames.loss: 19.0,
            SequenceExampleFeatureNames.horizon: 10
        },
    ]
    data_eval = [
        {
            SequenceExampleFeatureNames.module_name: 'module_1',
            SequenceExampleFeatureNames.loss: 30.0,
            SequenceExampleFeatureNames.horizon: 10
        },
        {
            SequenceExampleFeatureNames.module_name: 'module_2',
            SequenceExampleFeatureNames.loss: 1.,
            SequenceExampleFeatureNames.horizon: 10
        },
        {
            SequenceExampleFeatureNames.module_name: 'module_3',
            SequenceExampleFeatureNames.loss: 19.0,
            SequenceExampleFeatureNames.horizon: 10
        },
    ]
    new_profile = train_weights.create_new_profile(data_comp, data_eval)
    expected_new_prof = [{
        SequenceExampleFeatureNames.module_name: 'module_1',
        SequenceExampleFeatureNames.loss: 30.0,
        SequenceExampleFeatureNames.horizon: 10,
        SequenceExampleFeatureNames.regret: -17.0,
        SequenceExampleFeatureNames.reward: 0.3617020507016913
    }, {
        SequenceExampleFeatureNames.module_name: 'module_2',
        SequenceExampleFeatureNames.loss: 1.0,
        SequenceExampleFeatureNames.horizon: 10,
        SequenceExampleFeatureNames.regret: 1.0,
        SequenceExampleFeatureNames.reward: -1 / 1e-5
    }, {
        SequenceExampleFeatureNames.module_name: 'module_3',
        SequenceExampleFeatureNames.loss: 19.0,
        SequenceExampleFeatureNames.horizon: 10,
        SequenceExampleFeatureNames.regret: 0.0,
        SequenceExampleFeatureNames.reward: -0.0
    }]
    for prof, expected_prof in zip(new_profile, expected_new_prof):
      self.assertAllClose(prof[SequenceExampleFeatureNames.loss],
                          expected_prof[SequenceExampleFeatureNames.loss])
      self.assertAllClose(prof[SequenceExampleFeatureNames.regret],
                          expected_prof[SequenceExampleFeatureNames.regret])
      self.assertAllClose(prof[SequenceExampleFeatureNames.reward],
                          expected_prof[SequenceExampleFeatureNames.reward])

  def test_update_weights(self):
    train_weights = TrainingWeights(partitions=(0.5, 10.0, 20.0))
    data_comp = [
        {
            SequenceExampleFeatureNames.module_name: 'module_1',
            SequenceExampleFeatureNames.loss: 47.0,
            SequenceExampleFeatureNames.horizon: 10
        },
        {
            SequenceExampleFeatureNames.module_name: 'module_2',
            SequenceExampleFeatureNames.loss: .0,
            SequenceExampleFeatureNames.horizon: 10
        },
        {
            SequenceExampleFeatureNames.module_name: 'module_3',
            SequenceExampleFeatureNames.loss: 19.0,
            SequenceExampleFeatureNames.horizon: 10
        },
    ]
    data_eval = [
        {
            SequenceExampleFeatureNames.module_name: 'module_1',
            SequenceExampleFeatureNames.loss: 30.0,
            SequenceExampleFeatureNames.horizon: 10
        },
        {
            SequenceExampleFeatureNames.module_name: 'module_2',
            SequenceExampleFeatureNames.loss: 1.,
            SequenceExampleFeatureNames.horizon: 10
        },
        {
            SequenceExampleFeatureNames.module_name: 'module_3',
            SequenceExampleFeatureNames.loss: 19.0,
            SequenceExampleFeatureNames.horizon: 10
        },
    ]
    new_weights = train_weights.update_weights(data_comp, data_eval)
    self.assertAllClose(new_weights,
                        [0.27346137, 0.17961589, 0.27346137, 0.27346137])


if __name__ == '__main__':
  multiprocessing.handle_main(functools.partial(app.run, tf.test.main))
