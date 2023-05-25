from absl import app
import tensorflow as tf
import tensorflow_text as text

import numpy as np

import os
from typing import Optional, List

from google3.third_party.ml_compiler_opt.compiler_opt.rl import feature_ops
from google3.third_party.ml_compiler_opt.compiler_opt.rl import registry
from google3.third_party.ml_compiler_opt.compiler_opt.rl import attention
from google3.third_party.ml_compiler_opt.compiler_opt.rl.regalloc import regalloc_network
from google3.third_party.ml_compiler_opt.compiler_opt.rl import policy_saver
from absl import flags
from absl import logging
import gin
# from google3.third_party.ml_compiler_opt.compiler_opt.rl import policy_saver


# Have one class for the Encoder
# Have one class for the full semisupervised model
# only save the Encoder
# Output heads:
#   - reward
#   - next state
#   -

flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.'
)

flags.DEFINE_string('trace', None, '')
flags.DEFINE_string('root_dir', None, '')

flags.DEFINE_string('tpu', None, 'BNS address for the TPU')


FLAGS = flags.FLAGS

_ACTION_KEY = 'action'
_CUR_STATE_KEY = 'cur_state'
_NEXT_STATE_KEY = 'next_state'
_NEXT_ACTION_KEY = 'next_action'
_MLM_KEY = 'mlm'

_INSTR_COUNT = 64

_MLM_IGNORE_TOKEN = -1
_MLM_MASK_TOKEN = 18000 - 1


class Model(tf.keras.Model):

  def __init__(
      self,
      encoder_network,
      state_head,
      action_head,
      next_state_head,
      next_action_head,
      mlm_head,
  ):
    super().__init__(name='Model')
    self._encoder_network = encoder_network
    self._state_head = state_head
    self._action_head = action_head
    self._next_state_head = next_state_head
    self._next_action_head = next_action_head
    self._mlm_head = mlm_head

  def call(self, inputs):
    # TODO:
    # 1) export the non-reduced tensor because need it for the MLM outputs
    # 2) when reducing, mask the tensors that don't matter
    #       for masking, I think I should set the relevant outputs to zero and also zero out the labels.
    # 3) figure out the correct masks when training, might need custom training loop or custom loss functions.
    observation = inputs['obs']
    action = tf.one_hot(inputs['action'], depth=33)[:, :, tf.newaxis]

    use_def_obs = {
        k: v for k, v in observation.items() if k.startswith('use_def_')
    }
    encoded_state_per_token, encoded_state = self._encoder_network(use_def_obs)

    encoded_state_with_action = tf.concat([encoded_state, action], axis=-1)

    state = self._state_head(encoded_state)
    action = self._action_head(encoded_state)
    next_state = self._next_state_head(encoded_state_with_action)
    next_action = self._next_action_head(encoded_state_with_action)
    mlm = self._mlm_head(encoded_state_per_token)
    return {
        _CUR_STATE_KEY: state,
        _ACTION_KEY: action,
        _NEXT_STATE_KEY: next_state,
        _NEXT_ACTION_KEY: next_action,
        _MLM_KEY: mlm,
    }


def action_loss(label, pred):
  # Use the current mask
  print(label)
  pass


def state_loss(label, pred):
  # Decide whether to use the intersection of the masks or just the current mask
  # Probably use the current mask
  print(label)
  pass


def masked_language_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossEntropy(
      from_logits=True, reduction='none'
  )
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
  return loss


def create_model(encoder_network):
  action_head = tf.keras.Sequential([
      tf.keras.layers.Dense(128),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dense(1),
  ])
  state_head = tf.keras.Sequential([
      tf.keras.layers.Dense(128),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dense(65),
  ])
  next_state_head = tf.keras.Sequential([
      tf.keras.layers.Dense(128),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dense(65),
  ])
  next_action_head = tf.keras.Sequential([
      tf.keras.layers.Dense(128),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dense(1),
  ])
  mlm = tf.keras.Sequential([
      tf.keras.layers.Dense(18000),
  ])
  return Model(
      encoder_network=encoder_network,
      action_head=action_head,
      state_head=state_head,
      next_state_head=next_state_head,
      next_action_head=next_action_head,
      mlm_head=mlm,
  )


def _create_parser_fn(time_step_spec, action_spec):
  context_features = {}
  sequence_features = {
      tensor_spec.name: tf.io.FixedLenSequenceFeature(
          shape=tensor_spec.shape, dtype=tensor_spec.dtype
      )
      for tensor_spec in time_step_spec.observation.values()
  }
  sequence_features[action_spec.name] = tf.io.FixedLenSequenceFeature(
      shape=action_spec.shape, dtype=action_spec.dtype
  )

  def _parser_fn(serialized_proto):
    with tf.name_scope('parse'):
      _, parsed_sequence = tf.io.parse_single_sequence_example(
          serialized_proto,
          context_features=context_features,
          sequence_features=sequence_features,
      )
      return parsed_sequence

  return _parser_fn


def _get_masked_input_and_labels(encoded_texts):
  # https://keras.io/examples/nlp/masked_language_modeling/
  # 15% BERT masking
  encoded_texts_shape = tf.shape(encoded_texts)
  inp_mask = tf.random.uniform(encoded_texts_shape) < 0.15
  inp_mask[encoded_texts <= 0] = False
  labels = _MLM_IGNORE_TOKEN * tf.ones(encoded_texts_shape, dtype=fp.int64)
  # Set labels for masked tokens
  labels[inp_mask] = encoded_texts[inp_mask]

  # Prepare input
  encoded_texts_masked = tf.identity(encoded_texts)
  # Set input to [MASK] which is the last token for the 90% of tokens
  # This means leaving 10% unchanged
  inp_mask_2mask = inp_mask & (tf.random.uniform(encoded_texts_shape) < 0.90)
  encoded_texts_masked[inp_mask_2mask] = _MLM_MASK_TOKEN

  # Set 10% to a random token
  inp_mask_2random = inp_mask_2mask & (
      tf.random.uniform(encoded_texts_shape) < 1 / 9
  )
  encoded_texts_masked[inp_mask_2random] = tf.random.uniform(
      encoded_texts_shape, 3, mask_token_id, dtype=tf.int64
  )

  # Prepare sample_weights to pass to .fit() method
  sample_weights = tf.ones(encoded_texts_shape)
  sample_weights[labels == -1] = 0

  # y_labels would be same as encoded_texts i.e input tokens
  y_labels = tf.identity(encoded_texts)

  return encoded_texts_masked, y_labels, sample_weights


_MAX_PREDICTIONS_PER_BATCH = 32
random_selector = text.RandomItemSelector(
    max_selections_per_batch=_MAX_PREDICTIONS_PER_BATCH,
    selection_rate=0.2,
    unselectable_ids=[0],
)
mask_values_chooser = text.MaskValuesChooser(18000, _MLM_MASK_TOKEN, 0.8)


def get_masked_input_and_labels(encoded_texts):
  masked_token_ids, masked_pos, masked_lm_ids = text.mask_language_model(
      tf.RaggedTensor.from_tensor(encoded_texts, padding=0),
      item_selector=random_selector,
      mask_values_chooser=mask_values_chooser,
  )
  # NEed to fix this
  # Produce tensor 0 to 32, tile it and produce tensr of indices, then flatten
  # Then can use it in the scatter like I want
  masked_pos = masked_pos.to_tensor(
      default_value=-1, shape=(33, _MAX_PREDICTIONS_PER_BATCH)
  )

  ii = tf.tile(
      tf.range(33, dtype=tf.int64)[:, tf.newaxis],
      [1, _MAX_PREDICTIONS_PER_BATCH],
  )
  masked_pos = tf.stack([ii, masked_pos], axis=-1)
  scatter_values = tf.where(masked_pos[:, :, 1] < 0, 0.0, 1.0)
  masked_pos = tf.where(
      masked_pos < 0, tf.constant(0, dtype=tf.int64), masked_pos
  )
  weights = tf.scatter_nd(
      masked_pos, scatter_values[:, :, tf.newaxis], (33, _INSTR_COUNT, 1)
  )
  return (
      masked_token_ids.to_tensor(default_value=0, shape=(33, _INSTR_COUNT)),
      encoded_texts,
      weights,
  )


def create_dataset_fn(
    time_step_spec, action_spec, batch_size, shift, preprocessing_layer_creator
):
  assert shift < 0
  files_buffer_size = 100
  num_readers = 10
  num_map_threads = 8
  shuffle_buffer_size = 256

  parser_fn = _create_parser_fn(time_step_spec, action_spec)

  def _roll_experience(seq_ex):
    # Use tf agents nest map fn here
    def _roll(atom):
      return tf.roll(atom, shift=shift, axis=0)

    def _cutoff(atom):
      return atom[:shift]

    seq_ex_roll = tf.nest.map_structure(_roll, seq_ex)
    seq_ex_roll = tf.nest.map_structure(_cutoff, seq_ex_roll)
    seq_ex = tf.nest.map_structure(_cutoff, seq_ex)
    return seq_ex, seq_ex_roll

  preprocessing_layers = tf.nest.map_structure(
      preprocessing_layer_creator, time_step_spec.observation
  )

  def split_experience(seq_ex, seq_ex_roll):
    obs = {k: seq_ex[k] for k in seq_ex if k != action_spec.name}
    action = seq_ex[action_spec.name]
    obs_roll = {k: seq_ex_roll[k] for k in seq_ex_roll if k != action_spec.name}
    action_roll = seq_ex_roll[action_spec.name]
    return {
        'obs': obs,
        'action': action,
        'obs_roll': obs_roll,
        'action_roll': action_roll,
    }

  def _preprocessing_layer(seq_ex):
    for layer_name, layer in preprocessing_layers.items():
      seq_ex[layer_name] = layer(seq_ex[layer_name])
    return seq_ex

  def preprocess_experience(obs_dict):
    obs_dict['obs_cur'] = _preprocessing_layer(obs_dict['obs'].copy())
    obs_dict['obs_cur'] = tf.concat(
        [
            v
            for k, v in obs_dict['obs_cur'].items()
            if not k.startswith('use_def_')
        ],
        axis=-1,
    )
    obs_dict['obs_roll'] = _preprocessing_layer(obs_dict['obs_roll'])
    obs_dict['obs_roll'] = tf.concat(
        [
            v
            for k, v in obs_dict['obs_roll'].items()
            if not k.startswith('use_def_')
        ],
        axis=-1,
    )
    return obs_dict

  def to_inputs_and_labels(obs_dict):
    inputs = {'obs': obs_dict['obs'], 'action': obs_dict['action']}
    mlm_input, mlm_label, mlm_weight = get_masked_input_and_labels(
        obs_dict['obs']['use_def_opcode'][:, :_INSTR_COUNT]
    )
    inputs['use_def_opcode'] = mlm_input

    labels = {
        _CUR_STATE_KEY: obs_dict['obs_cur'],
        _ACTION_KEY: tf.expand_dims(obs_dict['action'], axis=-1),
        _NEXT_STATE_KEY: obs_dict['obs_roll'],
        _NEXT_ACTION_KEY: tf.expand_dims(obs_dict['action_roll'], axis=-1),
        _MLM_KEY: mlm_label,
    }
    mask = obs_dict['obs']['mask']
    sample_weights = {
        _CUR_STATE_KEY: mask,
        _ACTION_KEY: mask,
        _NEXT_STATE_KEY: None,
        _NEXT_ACTION_KEY: None,
        _MLM_KEY: mlm_weight,
    }
    return (inputs, labels, sample_weights)

  def _file_dataset_fn(data_path):
    return (
        tf.data.Dataset.list_files(data_path)
        .shuffle(files_buffer_size)
        .interleave(
            tf.data.TFRecordDataset,
            cycle_length=num_readers,
            block_length=1,
        )
        .filter(lambda string: tf.strings.length(string) > 0)
        .map(parser_fn, num_parallel_calls=num_map_threads)
        .map(_roll_experience, num_parallel_calls=num_map_threads)
        .map(split_experience, num_parallel_calls=num_map_threads)
        .map(preprocess_experience, num_parallel_calls=num_map_threads)
        .unbatch()
        .shuffle(shuffle_buffer_size)
        .map(to_inputs_and_labels, num_parallel_calls=num_map_threads)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

  return _file_dataset_fn


def get_strategy():
  if FLAGS.tpu:
    logging.info('Using TPU strategy.')
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    return tf.distribute.TPUStrategy(resolver)
  logging.info('Using CPU strategy.')
  return tf.distribute.get_strategy()


class SaveEncoderCallback(tf.keras.callbacks.Callback):

  def __init__(self, encoder, path):
    self._encoder = encoder
    self._path = path

  def on_epoch_end(self, epoch, logs=None):
    sm_path = os.path.join(self._path, f'epoch{epoch}')
    tflite_path = os.path.join(sm_path, policy_saver.TFLITE_MODEL_NAME)
    self._encoder.save(sm_path)
    policy_saver.convert_saved_model(sm_path, tflite_path)


def main(_):
  gin.parse_config_files_and_bindings(
      FLAGS.gin_files, bindings=None, skip_unknown=False
  )
  logging.info(gin.config_str())

  problem_config = registry.get_configuration()
  time_step_spec, action_spec = problem_config.get_signature_spec()
  preprocessing_layer_creator = problem_config.get_preprocessing_layer_creator()
  dataset_fn = create_dataset_fn(
      time_step_spec,
      action_spec,
      batch_size=256,
      shift=-1,
      preprocessing_layer_creator=preprocessing_layer_creator,
  )

  # inputs, labels, weights = next(iter(dataset))
  # import sys
  # sys.exit()

  strategy = get_strategy()
  with strategy.scope():
    logging.info('Creating model.')
    encoder_network = regalloc_network.InstructionEncoderNetwork(
        preprocessing_layer_creator
    )
    model = create_model(encoder_network)

    logging.info('Compiling model.')

    opt = tf.keras.optimizers.Adam(global_clipnorm=1.0)

    model.compile(
        optimizer=opt,
        loss={
            _CUR_STATE_KEY: 'mse',
            _ACTION_KEY: tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            _NEXT_STATE_KEY: 'mse',
            _NEXT_ACTION_KEY: tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            _MLM_KEY: tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
        },
        loss_weights={
            _ACTION_KEY: 1,
            _NEXT_STATE_KEY: 1,
            _NEXT_ACTION_KEY: 1,
            _MLM_KEY: 10,
        },
        metrics={
            _CUR_STATE_KEY: [],
            _ACTION_KEY: ['accuracy'],
            _NEXT_STATE_KEY: [],
            _NEXT_ACTION_KEY: ['accuracy'],
            _MLM_KEY: ['accuracy'],
        },
    )

  logging.info('Creating dataset.')
  dataset = dataset_fn(FLAGS.trace)
  tb = tf.keras.callbacks.TensorBoard(
      log_dir=FLAGS.root_dir,
      histogram_freq=0,
      write_graph=True,
      write_steps_per_second=True,
      update_freq='batch',
  )

  policy_dir = os.path.join(FLAGS.root_dir, 'policy')
  checkpoint = tf.keras.callbacks.ModelCheckpoint(
      filepath=policy_dir, save_weights_only=False, save_freq=1024
  )

  encoder_dir = os.path.join(FLAGS.root_dir, 'encoder')
  logging.info('Saving the encoder to %s', encoder_dir)
  encoder_saver = SaveEncoderCallback(encoder=encoder_network, path=encoder_dir)

  logging.info('Starting training.')
  model.fit(dataset, epochs=8, callbacks=[tb, checkpoint, encoder_saver])

  logging.info('Training complete.')


if __name__ == '__main__':
  app.run(main)
