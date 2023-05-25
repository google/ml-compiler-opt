import gin
import tensorflow as tf
from tf_agents.utils import nest_utils
from google3.third_party.ml_compiler_opt.compiler_opt.rl.regalloc.lr_encoder import config
from google3.third_party.ml_compiler_opt.compiler_opt.rl import attention


class MultiHeadEncoderModel(tf.keras.Model):

  def __init__(self, encoder, heads):
    super().__init__(name='MultiHeadEncoderModel')

    self._encoder = encoder
    self._heads = heads

  def get_encoder(self):
    return self._encoder

  def call(self, inputs, *args):
    observation = inputs['obs']
    action = tf.one_hot(inputs['action'], depth=config._NUM_REGISTERS)[
        :, :, tf.newaxis
    ]

    use_def_obs = {
        k: v
        for k, v in observation.items()
        if k.startswith(config._ENCODER_FEATURE_PREFIX)
    }
    encoded_state_per_token, encoded_state = self._encoder(use_def_obs)
    encoded_state_with_action = tf.concat([encoded_state, action], axis=-1)

    def get_input(head_name):
      if head_name == 'mlm':
        return encoded_state_per_token
      if head_name.startswith('next_'):
        return encoded_state_with_action
      return encoded_state

    outputs = {}
    for name in self._heads:
      head = self._heads[name]
      head_input = get_input(name)
      outputs[name] = head(head_input)

    return outputs


@gin.configurable
class LiveRangeEncoder(tf.keras.Model):

  def __init__(
      self,
      input_specs,
      preprocessing_layer_creator,
      *,
      num_layers=2,
      num_heads=2,
      model_dim=32,
      fcn_dim=128,
      num_extra_features=10,
  ):
    super().__init__(name='LiveRangeEncoder')
    self._encoder = attention.TransformerClassifier(
        num_tokens=config._OPCODE_VOCAB_SIZE,
        num_layers=num_layers,
        num_heads=num_heads,
        model_dim=model_dim,
        fcn_dim=fcn_dim,
        num_extra_features=num_extra_features,
    )
    self._linear_reshape = tf.keras.layers.Dense(config._ENCODING_SIZE)

    self._preprocessing_layers = {}
    for key in input_specs:
      self._preprocessing_layers[key] = preprocessing_layer_creator(key)

    def preprocessor(observations):
      pp_inputs = []
      for key in input_specs:
        pp_inputs.append(self._preprocessing_layers[key](observations[key]))
      return tf.concat(pp_inputs, axis=-1)

    self._preprocessor = preprocessor

  def call(self, observations):
    extra_hidden_state = self._preprocessor(observations)
    tokens = observations[config._OPCODE_KEY]
    encoded_state_per_token, encoded_state = self._encoder(
        tokens, extra_hidden_state
    )
    encoded_state = self._linear_reshape(encoded_state)
    return encoded_state_per_token, encoded_state


def create_model(
    input_specs: dict,
    output_specs: dict,
    preprocessing_layer_creator: dict,
    *,
    mlp_width: int = 128,
):
  encoder = LiveRangeEncoder(input_specs, preprocessing_layer_creator)

  output_heads = {}
  for name, spec in output_specs.items():
    size = spec.shape[-1]
    output_heads[name] = tf.keras.Sequential([
        tf.keras.layers.Dense(mlp_width),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(size),
    ])

  return MultiHeadEncoderModel(encoder=encoder, heads=output_heads)
