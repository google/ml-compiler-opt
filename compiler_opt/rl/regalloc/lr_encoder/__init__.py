"""Implementation of the 'lr_encoder' problem."""

import gin

from google3.third_party.ml_compiler_opt.compiler_opt.rl import problem_configuration
from google3.third_party.ml_compiler_opt.compiler_opt.rl.regalloc.lr_encoder import config
from google3.third_party.ml_compiler_opt.compiler_opt.rl.regalloc.lr_encoder import lr_encoder_runner


@gin.register(module='configs')
class LREncoderConfig(problem_configuration.ProblemConfiguration):
  """Expose the LR encoder configuration."""

  def get_runner_type(self):
    return lr_encoder_runner.LREncoderRunner

  def get_signature_spec(self):
    return config.get_lr_encoder_signature_spec()

  def get_preprocessing_layer_creator(self):
    return config.get_preprocessing_layer_creator()

  def get_nonnormalized_features(self):
    return config.get_nonnormalized_features()
