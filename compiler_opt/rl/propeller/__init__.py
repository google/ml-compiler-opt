"""Propeller RL configuration."""

import gin
import tensorflow as tf

from .. import problem_configuration
from . import config
from . import propeller_runner
from . import agent_config


@gin.register(module='configs')
class PropellerConfig(problem_configuration.ProblemConfiguration):
  """Propeller configuration for Regression RL."""

  def get_env(self):
    raise NotImplementedError(
        'get_env not implemented for RegressionPropellerConfig'
    )

  def get_runner_type(self):
    return propeller_runner.PropellerRunner

  def get_signature_spec(self):
    return config.get_propeller_regression_signature_spec()

  def get_preprocessing_layer_creator(self):
    return config.get_observation_processing_layer_creator()

  def get_nonnormalized_features(self):
    return config.get_nonnormalized_features()


@gin.register(module='configs')
class RegressionPropellerConfig(problem_configuration.ProblemConfiguration):
  """Propeller configuration for Regression RL."""

  def get_env(self):
    raise NotImplementedError(
        'get_env not implemented for RegressionPropellerConfig'
    )

  def get_runner_type(self):
    return propeller_runner.PropellerRunner

  def get_signature_spec(self):
    return config.get_propeller_regression_signature_spec()

  def get_preprocessing_layer_creator(self):
    return config.get_observation_processing_layer_creator()

  def get_nonnormalized_features(self):
    return config.get_nonnormalized_features()
