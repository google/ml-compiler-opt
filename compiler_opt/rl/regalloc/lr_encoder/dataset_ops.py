import tensorflow as tf
import tensorflow_text as text

from google3.third_party.ml_compiler_opt.compiler_opt.rl.regalloc.lr_encoder import config

_MAX_PREDICTIONS_PER_BATCH = 128

_PAD_TOKEN = 0
_MLM_IGNORE_TOKEN = -1
_MLM_MASK_TOKEN = 18000 - 1


def _get_masked_language_fn(*, selection_rate=0.33):
  random_selector = text.RandomItemSelector(
      max_selections_per_batch=_MAX_PREDICTIONS_PER_BATCH,
      selection_rate=0.2,
      unselectable_ids=[_PAD_TOKEN],
  )
  mask_values_chooser = text.MaskValuesChooser(
      config._OPCODE_VOCAB_SIZE, _MLM_MASK_TOKEN, 0.8
  )

  def fn(opcodes):
    masked_token_ids, masked_pos, masked_lm_ids = text.mask_language_model(
        tf.RaggedTensor.from_tensor(opcodes, padding=_PAD_TOKEN),
        item_selector=random_selector,
        mask_values_chooser=mask_values_chooser,
    )

    masked_pos = masked_pos.to_tensor(
        default_value=-1,
        shape=(config._NUM_REGISTERS, _MAX_PREDICTIONS_PER_BATCH),
    )

    ii = tf.tile(
        tf.range(config._NUM_REGISTERS, dtype=tf.int64)[:, tf.newaxis],
        [1, _MAX_PREDICTIONS_PER_BATCH],
    )
    masked_pos = tf.stack([ii, masked_pos], axis=-1)
    scatter_values = tf.where(masked_pos[:, :, 1] < 0, 0.0, 1.0)
    masked_pos = tf.where(
        masked_pos < 0, tf.constant(_PAD_TOKEN, dtype=tf.int64), masked_pos
    )
    weights = tf.scatter_nd(
        masked_pos,
        scatter_values[:, :, tf.newaxis],
        (config._NUM_REGISTERS, config._NUM_INSTRUCTIONS, 1),
    )
    return (
        masked_token_ids.to_tensor(
            default_value=0,
            shape=(config._NUM_REGISTERS, config._NUM_INSTRUCTIONS),
        ),
        opcodes,
        weights,
    )

  return fn


def _roll_experience(seq_ex, *, shift=-1):
  def _roll(atom):
    return tf.roll(atom, shift=shift, axis=0)

  def _cutoff(atom):
    return atom[:shift]

  seq_ex_roll = tf.nest.map_structure(_roll, seq_ex)
  seq_ex_roll = tf.nest.map_structure(_cutoff, seq_ex_roll)
  seq_ex = tf.nest.map_structure(_cutoff, seq_ex)
  return seq_ex, seq_ex_roll


def _split_sequence_example(seq_ex, seq_ex_roll):
  action_name = 'index_to_evict'
  obs = {k: seq_ex[k] for k in seq_ex if k != action_name}
  action = seq_ex[action_name]
  obs_roll = {k: seq_ex_roll[k] for k in seq_ex_roll if k != action_name}
  action_roll = seq_ex_roll[action_name]
  return {
      'obs': obs,
      'action': action,
      'obs_roll': obs_roll,
      'action_roll': action_roll,
  }


def _get_state_preprocessing_layer(
    regalloc_input_spec, preprocessing_layer_creator
):
  preprocessing_layers = {}
  for name, spec in regalloc_input_spec.items():
    preprocessing_layers[name] = preprocessing_layer_creator(spec)

  def _preprocessing_layer(seq_ex):
    pp = []
    for layer_name, layer in preprocessing_layers.items():
      pp.append(layer(seq_ex[layer_name]))
    return tf.concat(pp, axis=-1)

  def _layer(obs_dict):
    obs_dict['obs'] = obs_dict['obs']
    obs_dict['obs_cur'] = _preprocessing_layer(obs_dict['obs'])
    obs_dict['obs_roll'] = _preprocessing_layer(obs_dict['obs_roll'])
    return obs_dict

  return _layer


def _get_to_inputs_and_labels_fn():
  masked_language_fn = _get_masked_language_fn()

  def fn(obs_dict):
    inputs = {'obs': obs_dict['obs'], 'action': obs_dict['action']}
    mlm_input, mlm_label, mlm_weight = masked_language_fn(
        obs_dict['obs']['lr_use_def_opcode'][:, : config._NUM_INSTRUCTIONS]
    )
    inputs['lr_use_def_opcode'] = mlm_input

    labels = {
        config._STATE_KEY: obs_dict['obs_cur'],
        config._ACTION_KEY: tf.expand_dims(obs_dict['action'], axis=-1),
        config._NEXT_STATE_KEY: obs_dict['obs_roll'],
        config._NEXT_ACTION_KEY: tf.expand_dims(
            obs_dict['action_roll'], axis=-1
        ),
        config._MLM_KEY: mlm_label,
    }
    mask = obs_dict['obs']['mask']
    sample_weights = {
        config._STATE_KEY: mask,
        config._ACTION_KEY: mask,
        config._NEXT_STATE_KEY: None,
        config._NEXT_ACTION_KEY: None,
        config._MLM_KEY: mlm_weight,
    }
    return (inputs, labels, sample_weights)

  return fn


def process_dataset(
    dataset, regalloc_input_spec, regalloc_preprocessing_layer_creator
):
  shuffle_buffer_size = 128
  num_map_threads = 128
  state_preprocessing_layer = _get_state_preprocessing_layer(
      regalloc_input_spec, regalloc_preprocessing_layer_creator
  )
  to_inputs_and_labels_fn = _get_to_inputs_and_labels_fn()
  return (
      dataset.map(_roll_experience, num_parallel_calls=num_map_threads)
      .map(_split_sequence_example, num_parallel_calls=num_map_threads)
      .map(state_preprocessing_layer, num_parallel_calls=num_map_threads)
      .unbatch()
      .shuffle(shuffle_buffer_size)
      .map(to_inputs_and_labels_fn, num_parallel_calls=num_map_threads)
  )
