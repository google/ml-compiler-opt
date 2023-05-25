import os

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf

from google3.third_party.ml_compiler_opt.compiler_opt.rl import policy_saver
from google3.third_party.ml_compiler_opt.compiler_opt.rl.regalloc import config as regalloc_config
from google3.third_party.ml_compiler_opt.compiler_opt.rl.regalloc.lr_encoder import dataset_ops
from google3.third_party.ml_compiler_opt.compiler_opt.rl.regalloc.lr_encoder import model
from google3.third_party.ml_compiler_opt.compiler_opt.rl.regalloc.lr_encoder import config as encoder_config
from google3.third_party.ml_compiler_opt.compiler_opt.rl import registry  # pylint: disable=unused-import


flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.'
)

flags.DEFINE_string('trace', None, '')
flags.DEFINE_string('root_dir', None, '')

flags.DEFINE_string('tpu', None, 'BNS address for the TPU')

FLAGS = flags.FLAGS


def _create_parser_fn(time_step_spec, action_spec):
  context_features = {}
  sequence_features = {
      tensor_spec.name: tf.io.FixedLenSequenceFeature(
          shape=tensor_spec.shape, dtype=tensor_spec.dtype
      )
      for tensor_spec in time_step_spec.values()
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


def get_dataset_creator_fn(
    batch_size,
    input_spec,
    action_spec,
    regalloc_input_spec,
    regalloc_preprocessing_layer_creator,
):
  files_buffer_size = 64
  num_map_threads = 128
  num_readers = 64
  parser_fn = _create_parser_fn(input_spec, action_spec)

  def _file_dataset_fn(data_path):
    dataset = (
        tf.data.Dataset.list_files(data_path)
        .shuffle(files_buffer_size)
        .interleave(
            tf.data.TFRecordDataset,
            cycle_length=num_readers,
            block_length=1,
        )
        .filter(lambda string: tf.strings.length(string) > 0)
        .map(parser_fn, num_parallel_calls=num_map_threads)
    )
    dataset = dataset_ops.process_dataset(
        dataset, regalloc_input_spec, regalloc_preprocessing_layer_creator
    )
    return dataset.batch(batch_size, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE
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

  lr_input_specs, lr_encoding_spec = (
      encoder_config.get_lr_encoder_signature_spec()
  )

  regalloc_time_step_spec, regalloc_action_spec = (
      regalloc_config.get_regalloc_signature_spec()
  )
  regalloc_preprocessing_creator = (
      regalloc_config.get_observation_processing_layer_creator()
  )

  dataset_fn = get_dataset_creator_fn(
      batch_size=64,
      input_spec=encoder_config.get_input_specs(),
      action_spec=regalloc_action_spec,
      regalloc_input_spec=regalloc_time_step_spec.observation,
      regalloc_preprocessing_layer_creator=regalloc_preprocessing_creator,
  )

  strategy = get_strategy()
  with strategy.scope():
    logging.info('Creating model.')
    lr_model = model.create_model(
        lr_input_specs,
        encoder_config.get_output_specs(regalloc_preprocessing_creator),
        encoder_config.get_preprocessing_layer_creator(),
    )

    logging.info('Compiling model.')
    loss, loss_weights = encoder_config.get_loss()
    lr_model.compile(
        optimizer=tf.keras.optimizers.Adam(global_clipnorm=1.0),
        loss=loss,
        loss_weights=loss_weights,
        metrics=encoder_config.get_metrics(),
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
  encoder_saver = SaveEncoderCallback(
      encoder=lr_model.get_encoder(), path=encoder_dir
  )

  logging.info('Starting training.')
  lr_model.fit(dataset, epochs=8, callbacks=[tb, checkpoint, encoder_saver])

  logging.info('Training complete.')


if __name__ == '__main__':
  app.run(main)
