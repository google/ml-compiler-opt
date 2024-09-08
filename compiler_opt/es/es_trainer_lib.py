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
"""Local ES trainer."""

import tempfile
from typing import Optional
from absl import flags, logging
import functools
import gin
import tensorflow as tf
import os
import shutil

from compiler_opt.distributed import worker
from compiler_opt.distributed.local import local_worker_manager
from compiler_opt.es import blackbox_optimizers
from compiler_opt.es import gradient_ascent_optimization_algorithms
from compiler_opt.es import blackbox_learner
from compiler_opt.es import policy_utils
from compiler_opt.rl import compilation_runner, policy_saver, trace_data_collector

POLICY_NAME = "policy"

FLAGS = flags.FLAGS

_BETA1 = flags.DEFINE_float("beta1", 0.9,
                            "Beta1 for ADAM gradient ascent optimizer.")
_BETA2 = flags.DEFINE_float("beta2", 0.999,
                            "Beta2 for ADAM gradient ascent optimizer.")
_GRAD_REG_ALPHA = flags.DEFINE_float(
    "grad_reg_alpha", 0.01,
    "Weight of regularization term in regression gradient.")
_GRAD_REG_TYPE = flags.DEFINE_string(
    "grad_reg_type", "ridge",
    "Regularization method to use with regression gradient.")
_GRADIENT_ASCENT_OPTIMIZER_TYPE = flags.DEFINE_string(
    "gradient_ascent_optimizer_type", 'adam',
    "Gradient ascent optimization algorithm: 'momentum' or 'adam'")
_MOMENTUM = flags.DEFINE_float(
    "momentum", 0.0, "Momentum for momentum gradient ascent optimizer.")
_OUTPUT_PATH = flags.DEFINE_string("output_path", "",
                                   "Path to write all output")
_PRETRAINED_POLICY_PATH = flags.DEFINE_string(
    "pretrained_policy_path", None,
    "The path of the pretrained policy. If not provided, it will \
        construct a new policy with randomly initialized weights.")

_CORPUS_DIR = '/usr/local/google/home/aidengrossman/opt_mlregalloc/corpus_subset'
_CLANG_PATH = '/usr/local/google/home/aidengrossman/opt_mlregalloc/clang'
_TRACE_PATH = '/usr/local/google/home/aidengrossman/opt_mlregalloc/bb_trace.pb'
_FUNCTION_INDEX_PATH = '/usr/local/google/home/aidengrossman/opt_mlregalloc/function_index.pb'
_BB_TRACE_MODEL_PATH = '/usr/local/google/home/aidengrossman/opt_mlregalloc/basic_block_trace_model'


class ESWorker(worker.Worker):

  def __init__(self, *, all_gin):
    gin.parse_config(all_gin)
    policy = policy_utils.create_actor_policy()
    saver = policy_saver.PolicySaver({POLICY_NAME: policy})
    self._template_dir = tempfile.mkdtemp()
    saver.save(self._template_dir)

    self._models_for_test_path = '/usr/local/google/home/aidengrossman/output_models/'

  def es_compile(self, params: list[float], baseline_score: float) -> float:
    with tempfile.TemporaryDirectory() as tempdir:
      smdir = os.path.join(tempdir, 'sm')
      my_model = tf.saved_model.load(
          os.path.join(self._template_dir, POLICY_NAME))
      policy_utils.set_vectorized_parameters_for_policy(my_model, params)
      tf.saved_model.save(my_model, smdir, signatures=my_model.signatures)
      tflitedir = os.path.join(tempdir, 'tflite')
      policy_saver.convert_saved_model(
          smdir, os.path.join(tflitedir, policy_saver.TFLITE_MODEL_NAME))
      tf.io.gfile.copy(
          os.path.join(self._template_dir, POLICY_NAME,
                       policy_saver.OUTPUT_SIGNATURE),
          os.path.join(tflitedir, policy_saver.OUTPUT_SIGNATURE))

      trace_data_collector.compile_corpus(
          _CORPUS_DIR, tempdir, _CLANG_PATH, tflitedir, thread_count=4)
      score = trace_data_collector.evaluate_compiled_corpus(
          tempdir, _TRACE_PATH, _FUNCTION_INDEX_PATH, _BB_TRACE_MODEL_PATH, 4)

      reward = compilation_runner._calculate_reward(score, baseline_score)
      print(reward)

      output_path = os.path.join(self._models_for_test_path,
                                 "model" + str(reward))
      if reward > 0 and not os.path.exists(output_path):
        shutil.copytree(tflitedir, output_path)
      return compilation_runner._calculate_reward(score, baseline_score)


@gin.configurable
def train(worker_class=None):
  """Train with ES."""

  # Create directories
  if not tf.io.gfile.isdir(_OUTPUT_PATH.value):
    tf.io.gfile.makedirs(_OUTPUT_PATH.value)

  # Construct the policy and upload it
  policy = policy_utils.create_actor_policy()
  saver = policy_saver.PolicySaver({POLICY_NAME: policy})

  # Save the policy
  policy_save_path = os.path.join(_OUTPUT_PATH.value, "policy")
  saver.save(policy_save_path)

  # Get initial parameter
  if not _PRETRAINED_POLICY_PATH.value:
    # Use randomly initialized parameters
    logging.info("Use random parameters")
    initial_parameters = policy_utils.get_vectorized_parameters_from_policy(
        policy)
    logging.info("Parameter dimension: %s", initial_parameters.shape)
    logging.info("Initial parameters: %s", initial_parameters)
  else:
    # Read the parameters from the pretrained policy
    logging.info("Reading policy parameters from %s",
                 _PRETRAINED_POLICY_PATH.value)
    # Load the policy
    pretrained_policy = tf.saved_model.load(_PRETRAINED_POLICY_PATH.value)
    initial_parameters = policy_utils.get_vectorized_parameters_from_policy(
        pretrained_policy)

  policy_parameter_dimension = (
      policy_utils.get_vectorized_parameters_from_policy(policy).shape[0])
  if policy_parameter_dimension != initial_parameters.shape[0]:
    raise ValueError("Pretrained policy dimension is incorrect")

  logging.info("Parameter dimension: %s", initial_parameters.shape)
  logging.info("Initial parameters: %s", initial_parameters)

  # Construct policy saver
  policy_saver_function = functools.partial(
      policy_utils.save_policy,
      policy=policy,
      save_folder=os.path.join(_OUTPUT_PATH.value, "saved_policies"))

  # Get learner config
  learner_config = blackbox_learner.BlackboxLearnerConfig()

  # the following are from Blackbox Library.
  init_current_input = initial_parameters
  init_iteration = 0
  metaparams = []  # Ignore meta params for state normalization for now
  # TODO(linzinan): delete all unused parameters.

  # ------------------ GRADIENT ASCENT OPTIMIZERS ------------------------------
  if _GRADIENT_ASCENT_OPTIMIZER_TYPE.value == "momentum":
    logging.info("Running momentum gradient ascent optimizer")
    # You can obtain a vanilla gradient ascent optimizer by setting momentum=0.0
    # and setting step_size to the desired learning rate.
    gradient_ascent_optimizer = (
        gradient_ascent_optimization_algorithms.MomentumOptimizer(
            learner_config.step_size, _MOMENTUM.value))
  elif _GRADIENT_ASCENT_OPTIMIZER_TYPE.value == "adam":
    logging.info("Running Adam gradient ascent optimizer")
    gradient_ascent_optimizer = (
        gradient_ascent_optimization_algorithms.AdamOptimizer(
            learner_config.step_size, _BETA1.value, _BETA2.value))
  else:
    logging.info("No gradient ascent \
                 optimizer selected. Stopping.")
    return
  # ----------------------------------------------------------------------------

  # ------------------ OPTIMIZERS ----------------------------------------------
  if learner_config.blackbox_optimizer == (
      blackbox_optimizers.Algorithm.MONTE_CARLO):
    logging.info("Running ES/ARS. Filtering: %s directions",
                 str(learner_config.num_top_directions))
    blackbox_optimizer = blackbox_optimizers.MonteCarloBlackboxOptimizer(
        learner_config.precision_parameter, learner_config.est_type,
        learner_config.fvalues_normalization,
        learner_config.hyperparameters_update_method, metaparams, None,
        learner_config.num_top_directions, gradient_ascent_optimizer)
  elif learner_config.blackbox_optimizer == (
      blackbox_optimizers.Algorithm.TRUST_REGION):
    logging.info("Running trust region")
    tr_params = {
        "init_radius": FLAGS.tr_init_radius,
        "grow_threshold": FLAGS.tr_grow_threshold,
        "grow_factor": FLAGS.tr_grow_factor,
        "shrink_neg_threshold": FLAGS.tr_shrink_neg_threshold,
        "shrink_factor": FLAGS.tr_shrink_factor,
        "reject_threshold": FLAGS.tr_reject_threshold,
        "reject_factor": FLAGS.tr_reject_factor,
        "dense_hessian": FLAGS.tr_dense_hessian,
        "sub_termination": FLAGS.tr_sub_termination,
        "subproblem_maxiter": FLAGS.tr_subproblem_maxiter,
        "minimum_radius": FLAGS.tr_minimum_radius,
        "grad_type": FLAGS.grad_type,
        "grad_reg_type": _GRAD_REG_TYPE.value,
        "grad_reg_alpha": _GRAD_REG_ALPHA.value
    }
    for param, value in tr_params.items():
      logging.info("%s: %s", param, value)
      blackbox_optimizer = blackbox_optimizers.TrustRegionOptimizer(
          learner_config.precision_parameter, learner_config.est_type,
          learner_config.fvalues_normalization,
          learner_config.hyperparameters_update_method, metaparams, tr_params)
  elif learner_config.blackbox_optimizer == (
      blackbox_optimizers.Algorithm.SKLEARN_REGRESSION):
    logging.info("Running Regression Based Optimizer")
    blackbox_optimizer = blackbox_optimizers.SklearnRegressionBlackboxOptimizer(
        _GRAD_REG_TYPE.value, _GRAD_REG_ALPHA.value, learner_config.est_type,
        learner_config.fvalues_normalization,
        learner_config.hyperparameters_update_method, metaparams, None,
        gradient_ascent_optimizer)
  else:
    raise ValueError(
        f"Unknown optimizer: '{learner_config.blackbox_optimizer}'")

  # Get baseline score
  with tempfile.TemporaryDirectory() as tempdir:
    trace_data_collector.compile_corpus(_CORPUS_DIR, tempdir, _CLANG_PATH)
    baseline_score = trace_data_collector.evaluate_compiled_corpus(
        tempdir, _TRACE_PATH, _FUNCTION_INDEX_PATH, _BB_TRACE_MODEL_PATH)

  logging.info("Initializing blackbox learner.")
  learner = blackbox_learner.BlackboxLearner(
      blackbox_opt=blackbox_optimizer,
      tf_policy_path=os.path.join(policy_save_path, POLICY_NAME),
      output_dir=_OUTPUT_PATH.value,
      policy_saver_fn=policy_saver_function,
      model_weights=init_current_input,
      config=learner_config,
      initial_step=init_iteration,
      baseline_score=baseline_score)

  if not worker_class:
    logging.info("No Worker class selected. Stopping.")
    return

  logging.info("Ready to train: running for %d steps.",
               learner_config.total_steps)

  with local_worker_manager.LocalWorkerPoolManager(
      ESWorker,
      learner_config.total_num_perturbations,
      all_gin=gin.config_str()) as pool:
    for _ in range(learner_config.total_steps):
      learner.run_step(pool)

  return learner.get_model_weights()
