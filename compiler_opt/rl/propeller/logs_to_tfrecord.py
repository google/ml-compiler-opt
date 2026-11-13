"""Tool to convert Propeller log files to TFRecord."""

import glob
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from .. import log_reader

flags.DEFINE_string('input_log_dir', None, 'Directory containing .log files.')
flags.DEFINE_string(
    'output_tfrecord', 'propeller.tfrecord', 'Output TFRecord file.'
)

FLAGS = flags.FLAGS


def main(_):
  log_files = glob.glob(os.path.join(FLAGS.input_log_dir, '*.log'))
  logging.info('Found %d log files.', len(log_files))

  with tf.io.TFRecordWriter(FLAGS.output_tfrecord) as writer:
    total_records = 0
    for log_file in log_files:
      logging.info('Processing %s', log_file)
      try:
        sequence_examples = log_reader.read_log_as_sequence_examples(log_file)
        for se in sequence_examples.values():
          writer.write(se.SerializeToString())
          total_records += 1
      except Exception as e:  # pylint: disable=broad-except
        logging.error('Error processing %s: %s', log_file, e)

  logging.info(
      'Done. Written %d records to %s', total_records, FLAGS.output_tfrecord
  )


if __name__ == '__main__':
  flags.mark_flag_as_required('input_log_dir')
  app.run(main)
