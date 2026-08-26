"""Tests for Propeller-specific Behavioral Cloning agent configuration."""

import gin
import tensorflow as tf
from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step

# agent config import

class PropellerBCAgentConfigTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gin.clear_config()

  def test_regression_agent_creation(self):
    # Define specs
    # Observation spec needs score_gain to simulate propeller data
    observation_spec = {
        'obs': tensor_spec.TensorSpec(shape=(10,), dtype=tf.float32, name='obs'),
        'score_gain': tensor_spec.TensorSpec(shape=(1,), dtype=tf.float32, name='score_gain'),
    }
    time_step_spec = time_step.time_step_spec(observation_spec)

    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(1,), dtype=tf.float32, minimum=-100.0, maximum=100.0, name='action'
    )

    gin.bind_parameter('BehavioralCloningAgent.optimizer',
                       tf.compat.v1.train.AdamOptimizer())

    # Create config
    config = agent_config.PropellerBCAgentConfig(
        time_step_spec=time_step_spec, action_spec=action_spec
    )

    # Dummy network returning 1D tensor matching action_spec
    class DummyNetwork(network.Network):

      def __init__(self, input_tensor_spec, action_spec, name=None, **kwargs):
        super().__init__(
            input_tensor_spec=input_tensor_spec, state_spec=(), name=name
        )

      def call(
          self, _observation, _step_type=None, network_state=(), _training=False
      ):
        # Output raw prediction matching action_spec
        return tf.constant([[5.0]]), network_state

    # Mock preprocessing layers
    # The target_1d extraction uses index 0 of score_gain
    # In actual code, preprocessing layer returns [bucketized, bucketized^2, ...]
    # We mock it returning a [Batch, 2] tensor
    preprocessing_layers = {
        'obs': tf.keras.layers.Lambda(lambda x: x),
        'score_gain': tf.keras.layers.Lambda(lambda x: tf.concat([x, x], axis=-1)), # Returns [Batch, 2]
    }

    # Create agent
    agent = config.create_agent(preprocessing_layers, DummyNetwork)

    self.assertIsInstance(
        agent, behavioral_cloning_agent.BehavioralCloningAgent
    )
    self.assertTrue(callable(agent._bc_loss_fn))


if __name__ == '__main__':
  tf.test.main()
