"""attention.py

Defines various building-blocks for attention in neural networks. Namely, the
transformer encoder, which is a sequence-to-sequence model which uses
self-attention to model relationships within the sequence.
"""

import tensorflow as tf
import numpy as np

from typing import Optional, List


def positional_encoding(length, depth):
  """Build a positional encoding tensor.

  Taken from https://www.tensorflow.org/text/tutorials/transformer.

  Args:
    length: the number of sin/cos samples to generate.
    depth: the depth of the embedding which the encoding will be summed with.

  Returns:
    A tensor of shape (length, depth) representing the positional encoding.
  """
  depth = depth / 2

  positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

  angle_rates = 1 / (10000**depths)  # (1, depth)
  angle_rads = positions * angle_rates  # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)], axis=-1
  )

  return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
  """A positional embedding layer.

  A "positional embedding" is a sum of a token embedding with a positional
  encoding, which is used as the initial layer of a transformer encoder
  network.

  Taken from https://www.tensorflow.org/text/tutorials/transformer.
  """

  def __init__(self, vocab_size, d_model):
    """Initialize the positional embedding.

    Args:
      vocab_size: the size of the vocab, which should be one more than the
        maximum token value which will be seen during training/inference.
      d_model: the dimension of the model (size of the embedding vector)
    """
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(
        vocab_size, d_model, mask_zero=True
    )
    self.pos_encoding = tf.constant(
        positional_encoding(length=2048, depth=d_model), dtype=tf.float32
    )

  def compute_mask(self, *args, **kwargs):
    """Returns a mask for the given input."""
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    """Perform the positional embedding."""
    length = tf.shape(x)[-1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x


class TransformerEncoderLayer(tf.keras.layers.Layer):
  """Transformer Encoder.

  See https://arxiv.org/abs/1706.03762 for more details.
  """

  def __init__(
      self,
      num_heads: int,
      model_dim: int,
      fcn_dim: int,
      attention_axes: Optional[List] = None,
  ):
    """Initialize the transformer encoder.

    Args:
      num_heads: number of distinct attention heads within the layer.
      model_dim: dimension of the model, also the dimension of the embedding.
      fcn_dim: dimension of the fully-connected layers between attention layers.
      attention_axes: which axes in the input tensors perform attention across.
    """
    super().__init__()
    self._mha = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=model_dim, attention_axes=attention_axes
    )
    self._mha_norm = tf.keras.layers.LayerNormalization()
    self._mha_add = tf.keras.layers.Add()

    self._fcn = tf.keras.Sequential([
        tf.keras.layers.Dense(fcn_dim),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(model_dim),
        tf.keras.layers.Dropout(0.1),
    ])
    self._fcn_norm = tf.keras.layers.LayerNormalization()
    self._fcn_add = tf.keras.layers.Add()

  def call(self, x, attention_mask=None):
    """Call the transformer encoder."""
    x_attended = self._mha(
        query=x, value=x, key=x, attention_mask=attention_mask
    )
    x = self._mha_add([x_attended, x])
    x = self._mha_norm(x)

    x_fcn = self._fcn(x)
    x = self._fcn_add([x_fcn, x])
    x = self._fcn_norm(x)

    return x


class TransformerClassifier(tf.keras.layers.Layer):

  def __init__(
      self,
      *,
      num_tokens: int,
      num_layers: int,
      num_heads: int,
      model_dim: int,
      fcn_dim: int,
      num_extra_features: int = 0,
  ):
    super().__init__()

    self._model_dim = model_dim
    self._ctx_token = tf.constant(num_tokens, dtype=tf.int64)
    self._embed_layer = PositionalEmbedding(
        num_tokens + 1, model_dim - num_extra_features
    )
    self._transformer_layers = [
        TransformerEncoderLayer(
            num_heads=num_heads,
            model_dim=model_dim,
            fcn_dim=fcn_dim,
            attention_axes=(2,),
        )
        for _ in range(num_layers)
    ]

  def __call__(self, x, extra_hidden_state):
    # [B, 33, 1 + I] --> [B, 33, 1 + I, E]
    mask = self._embed_layer.compute_mask(x)
    x = self._embed_layer(x)

    # Append the extra hidden state
    x = tf.concat([x, extra_hidden_state], axis=-1)

    mask1 = mask[:, :, :, tf.newaxis]
    mask2 = mask[:, :, tf.newaxis, :]
    attn_mask = tf.cast(mask1, dtype=tf.int64) + tf.cast(mask2, dtype=tf.int64)
    attn_mask = attn_mask > 0

    for transformer_layer in self._transformer_layers:
      x = transformer_layer(x, attention_mask=attn_mask)

    mask_reduce = tf.cast(mask[:, :, :, tf.newaxis], dtype=tf.float32)
    x_reduced = (tf.reduce_sum(mask_reduce * x, axis=-2)) / (
        tf.reduce_sum(mask_reduce, axis=-2) + 1e-3
    )
    return x, x_reduced
