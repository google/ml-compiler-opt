"""Propeller Training config."""

import gin
import tensorflow as tf
from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.networks import network
from tf_agents.policies import actor_policy
from tf_agents.specs import tensor_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.trajectories import time_step as ts
from .. import agent_config
from .. import feature_ops


@gin.configurable()
def get_propeller_regression_signature_spec():
  """Returns (time_step_spec, action_spec) for Propeller Regression."""
  observation_spec = {
      key: tf.TensorSpec(dtype=tf.float32, shape=(1,), name=key)
      for key in (
          'unsplit_density',
          'split_density',
          'unsplit_size',
          'unsplit_freq',
          'split_size',
          'split_freq',
          'score_gain',
          'edge1_weight',
          'edge2_weight',
          'edge1_distance',
          'edge2_distance',
          'broken_bond_weight',
          'broken_bond_distance',
      )
  }

  #   observation_spec['edge1_type'] = tf.TensorSpec(
  #       dtype=tf.float32, shape=(4,), name='edge1_type'
  #   )
  #   observation_spec['edge2_type'] = tf.TensorSpec(
  #       dtype=tf.float32, shape=(4,), name='edge2_type'
  #   )
  #   observation_spec['broken_bond_type'] = tf.TensorSpec(
  #       dtype=tf.float32, shape=(4,), name='broken_bond_type'
  #   )

  observation_spec['split_s1_is_entry'] = tf.TensorSpec(
      dtype=tf.int64, shape=(1,), name='split_s1_is_entry'
  )
  observation_spec['split_s2_is_entry'] = tf.TensorSpec(
      dtype=tf.int64, shape=(1,), name='split_s2_is_entry'
  )
  observation_spec['unsplit_is_entry'] = tf.TensorSpec(
      dtype=tf.int64, shape=(1,), name='unsplit_is_entry'
  )
  #   observation_spec['decision_id'] = tf.TensorSpec(
  #       dtype=tf.int64, shape=(1,), name='decision_id'
  #   )

  reward_spec = tf.TensorSpec(dtype=tf.float32, shape=(), name='reward')
  time_step_spec = time_step.time_step_spec(observation_spec, reward_spec)

  # Target action is a single scalar value.
  action_spec = tensor_spec.BoundedTensorSpec(
      dtype=tf.float32,
      shape=(1,),
      minimum=-1e9,  # Changed to allow symmetric log negative values
      maximum=1e9,
      name='target_score_gain',
  )

  return time_step_spec, action_spec


@gin.configurable
def log1p_preprocessing(x):
  """Safely applies log(1+x) to input features."""
  # Cast to float32 and use tf.maximum to prevent any accidental negative
  # values from producing NaNs during the log operation.
  safe_x = tf.maximum(tf.cast(x, tf.float32), 0.0)
  return tf.math.log1p(safe_x)


@gin.configurable
def get_observation_processing_layer_creator(
    quantile_file_dir=None,
    with_sqrt=True,
    with_z_score_normalization=True,
    eps=1e-8,
):
  """Wrapper for observation_processing_layer."""
  quantile_map = feature_ops.build_quantile_map(quantile_file_dir)

  def observation_processing_layer(obs_spec):
    """Creates the layer to process observation given obs_spec."""
    if obs_spec.name == 'decision_id':
      return tf.keras.layers.Lambda(lambda x: x)

    if obs_spec.name in ['edge1_type', 'edge2_type', 'broken_bond_type']:
      tf.print(
          'INFO: Log1p preprocessing for feature:',
          obs_spec.name,
      )
      return tf.keras.layers.Lambda(log1p_preprocessing)

    if obs_spec.name in get_onehot_features():
      tf.print(
          'INFO: One-hot feature, skipping normalization:',
          obs_spec.name,
      )
      return tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))

    elif obs_spec.name in get_nonnormalized_features():
      tf.print(
          'INFO: Non-normalized feature, skipping normalization:',
          obs_spec.name,
      )
      return tf.keras.layers.Lambda(feature_ops.identity_fn)

    if obs_spec.name not in quantile_map:
      tf.print(
          'WARNING: Missing quantile for feature, skipping normalization:',
          obs_spec.name,
      )
      return tf.keras.layers.Lambda(feature_ops.identity_fn)

    tf.print(
        'INFO: Normalizing feature:',
        obs_spec.name,
    )
    quantile = quantile_map[obs_spec.name]
    return tf.keras.layers.Lambda(
        feature_ops.get_normalize_fn(
            quantile,
            with_sqrt=False,
            with_z_score_normalization=False,
            eps=eps,
            preprocessing_fn=log1p_preprocessing,
        )
    )

  return observation_processing_layer


@gin.configurable()
def get_onehot_features():
  return ['merge_order']


@gin.configurable()
def get_nonnormalized_features():
  return [
      'reward',
      # 'is_chosen',
      'decision_id',
      'split_s1_is_entry',
      'split_s2_is_entry',
      'unsplit_is_entry',
      'merge_order',
      'mask',
  ]
