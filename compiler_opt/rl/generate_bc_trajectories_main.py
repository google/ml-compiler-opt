import functools
from absl import app
from absl import flags
from absl import logging
import gin

from compiler_opt.rl import generate_bc_trajectories
from compiler_opt.tools import generate_test_model  # pylint:disable=unused-import

from tf_agents.system import system_multiprocessing as multiprocessing

flags.FLAGS['gin_files'].allow_override = True
flags.FLAGS['gin_bindings'].allow_override = True

FLAGS = flags.FLAGS

def main(_):
  gin.parse_config_files_and_bindings(
      FLAGS.gin_files, bindings=FLAGS.gin_bindings, skip_unknown=True)
  logging.info(gin.config_str())

  generate_bc_trajectories.gen_trajectories()


if __name__ == '__main__':
  multiprocessing.handle_main(functools.partial(app.run, main))
