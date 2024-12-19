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
"""Module for collect data of inlining-for-size."""

import gin
from typing import Type

import numpy as np
import tensorflow as tf
from tf_agents.trajectories import time_step

from compiler_opt.rl.inlining import config
from compiler_opt.rl.inlining import env


@gin.register
def get_inlining_signature_spec():
  """Returns (time_step_spec, action_spec) for collecting IL trajectories."""
  time_step_spec, action_spec = config.get_inlining_signature_spec()
  observation_spec = time_step_spec.observation
  observation_spec.update(
      dict((key, tf.TensorSpec(dtype=tf.int64, shape=(), name=key)) for key in (
          'is_callee_avail_external',
          'is_caller_avail_external',
          # inlining_default is not used as feature in training.
          'inlining_default')))

  time_step_spec = time_step.time_step_spec(observation_spec,
                                            time_step_spec.reward)

  return time_step_spec, action_spec


@gin.register
def get_task_type() -> Type[env.InliningForSizeTask]:
  """Returns the task type for the trajectory collection."""
  return env.InliningForSizeTask


@gin.register
def greedy_policy(state: time_step.TimeStep):
  """Greedy policy playing the inlining_default action."""
  return np.array(state.observation['inlining_default'])


@gin.register
def explore_on_avail_external(state_observation: tf.Tensor) -> bool:
  return state_observation.numpy()[0]
