"""LR encoder training config."""

import gin
import tensorflow as tf
import tf_agents
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from google3.third_party.ml_compiler_opt.compiler_opt.rl import feature_ops
from google3.third_party.ml_compiler_opt.compiler_opt.rl.regalloc import config

_NUM_REGISTERS = 33
_NUM_INSTRUCTIONS = 64
_OPCODE_VOCAB_SIZE = 20000
_ENCODING_SIZE = 16
_OPCODE_KEY = 'lr_use_def_opcode'

_ENCODER_FEATURE_PREFIX = 'lr_use_def'

_ACTION_KEY = 'action'
_STATE_KEY = 'state'
_NEXT_STATE_KEY = 'next_state'
_NEXT_ACTION_KEY = 'next_action'
_MLM_KEY = 'mlm'


@gin.configurable
def get_lr_encoder_signature_spec():
  observation_spec = {}
  observation_spec[_OPCODE_KEY] = tensor_spec.BoundedTensorSpec(
      dtype=tf.int64,
      shape=(_NUM_REGISTERS, _NUM_INSTRUCTIONS),
      name=_OPCODE_KEY,
      minimum=0,
      maximum=_OPCODE_VOCAB_SIZE,
  )
  observation_spec['lr_use_def_read'] = tensor_spec.BoundedTensorSpec(
      dtype=tf.int64,
      shape=(_NUM_REGISTERS, _NUM_INSTRUCTIONS),
      name='lr_use_def_read',
      minimum=0,
      maximum=1,
  )
  observation_spec['lr_use_def_write'] = tensor_spec.BoundedTensorSpec(
      dtype=tf.int64,
      shape=(_NUM_REGISTERS, _NUM_INSTRUCTIONS),
      name='lr_use_def_write',
      minimum=0,
      maximum=1,
  )
  observation_spec.update(
      {
          name: tensor_spec.BoundedTensorSpec(
              dtype=tf.int64,
              shape=(_NUM_REGISTERS, _NUM_INSTRUCTIONS),
              name=name,
              minimum=0,
              maximum=1,
          )
          for name in [
              'lr_use_def_is_use',
              'lr_use_def_is_def',
              'lr_use_def_is_implicit',
              'lr_use_def_is_renamable',
              'lr_use_def_is_ind_var_update',
              'lr_use_def_is_hint',
          ]
      }
  )
  observation_spec['lr_use_def_freq'] = tf.TensorSpec(
      dtype=tf.float32,
      shape=(_NUM_REGISTERS, _NUM_INSTRUCTIONS),
      name='lr_use_def_freq',
  )

  encoding_spec = tf.TensorSpec(
      dtype=tf.float32, shape=(33, _ENCODING_SIZE), name='lr_encoding'
  )

  return observation_spec, encoding_spec


def get_input_specs():
  encoder_observation_spec, encoding_spec = get_lr_encoder_signature_spec()
  del encoding_spec

  regalloc_time_step_spec, regalloc_action_spec = (
      config.get_regalloc_signature_spec()
  )

  # Ensure that there are no overlaps in feature names between the encoder and the RL problem
  common_keys = (
      encoder_observation_spec.keys()
      & regalloc_time_step_spec.observation.keys()
  )
  assert len(common_keys) == 0

  supervised_input_spec = {
      **encoder_observation_spec,
      **regalloc_time_step_spec.observation,
  }

  return supervised_input_spec


def get_output_specs(regalloc_preprocessing_layer_creator):
  regalloc_time_step_spec, regalloc_action_spec = (
      config.get_regalloc_signature_spec()
  )
  del regalloc_action_spec

  random_regalloc_state = tf_agents.specs.sample_spec_nest(
      regalloc_time_step_spec.observation
  )
  for key in random_regalloc_state:
    preprocessing_layer = regalloc_preprocessing_layer_creator(
        regalloc_time_step_spec.observation[key]
    )
    random_regalloc_state[key] = preprocessing_layer(
        tf.expand_dims(random_regalloc_state[key], axis=0)
    )
  random_regalloc_state = tf.concat(
      list(random_regalloc_state.values()), axis=-1
  )
  random_regalloc_state = tf.squeeze(random_regalloc_state, axis=0)
  regalloc_state_shape = random_regalloc_state.shape

  action_spec = tf.TensorSpec(
      dtype=tf.float32, shape=(_NUM_REGISTERS, 1), name=_ACTION_KEY
  )
  state_spec = tf.TensorSpec(
      dtype=tf.float32, shape=regalloc_state_shape, name=_STATE_KEY
  )
  mlm_spec = tf.TensorSpec(
      dtype=tf.float32,
      shape=(_NUM_REGISTERS, _NUM_INSTRUCTIONS, _OPCODE_VOCAB_SIZE),
      name=_MLM_KEY,
  )
  return {
      _ACTION_KEY: action_spec,
      _STATE_KEY: state_spec,
      _NEXT_ACTION_KEY: action_spec,
      _NEXT_STATE_KEY: state_spec,
      _MLM_KEY: mlm_spec,
  }


def get_loss():
  loss_fns = {
      _STATE_KEY: 'mse',
      _ACTION_KEY: tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True
      ),
      _NEXT_STATE_KEY: 'mse',
      _NEXT_ACTION_KEY: tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True
      ),
      _MLM_KEY: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  }
  loss_weights = {
      _ACTION_KEY: 1.0,
      _STATE_KEY: 1.0,
      _NEXT_ACTION_KEY: 1.0,
      _NEXT_STATE_KEY: 1.0,
      _MLM_KEY: 10.0,
  }
  return loss_fns, loss_weights


def get_metrics():
  return {
      _ACTION_KEY: ['accuracy'],
      _STATE_KEY: [],
      _NEXT_ACTION_KEY: ['accuracy'],
      _NEXT_STATE_KEY: [],
      _MLM_KEY: ['accuracy'],
  }


@gin.configurable
def get_preprocessing_layer_creator(
    quantile_file_dir='/cns/oz-d/home/mlcompileropt-dev/regalloc-transformer/vocab',
    with_sqrt=True,
    with_z_score_normalization=True,
    eps=1e-8,
):
  """Wrapper for observation_processing_layer."""
  quantile_map = feature_ops.build_quantile_map(quantile_file_dir)

  def preprocessing_layer_creator(obs_spec):
    """Creates the layer to process observation given obs_spec."""
    if obs_spec.name == 'lr_use_def_freq':
      quantile = quantile_map[obs_spec.name]
      first_non_zero = 0
      for x in quantile:
        if x > 0:
          first_non_zero = x
          break

      normalize_fn = feature_ops.get_normalize_fn(
          quantile, with_sqrt, with_z_score_normalization, eps
      )
      return tf.keras.layers.Lambda(normalize_fn)
    return tf.keras.layers.Lambda(feature_ops.identity_fn)

  return preprocessing_layer_creator


def get_nonnormalized_features():
  return [
      'lr_use_def_opcode',
      'lr_use_def_read',
      'lr_use_def_write',
      'lr_use_def_is_use',
      'lr_use_def_is_def',
      'lr_use_def_is_implicit',
      'lr_use_def_is_renamable',
      'lr_use_def_is_ind_var_update',
      'lr_use_def_is_hint',
  ]
