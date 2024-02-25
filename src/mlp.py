"""Fair MLP tree."""

import tensorflow as tf
import tensorflow_probability as tfp


class FairReLUNetwork(tf.Module):
  """Fair MLP."""
  def __init__(self,
               data_dim,
               tree_depth,
               num_classes,
               activation='relu'):
    """Constructor.
    
    Args:
      data_dim: dimension of the data.
      tree_depth: depth of the binary tree.
      num_classes: number of target task classes.
      activation: activation function.
    """
    super(FairReLUNetwork, self).__init__()
    assert tree_depth > 1
    self.num_internal_nodes = 2**tree_depth - 1
    # internal node parameters
    self.weight = tf.Variable(
        tf.random.normal([data_dim, self.num_internal_nodes]), name='W'
    )
    self.bias = tf.Variable(
        tf.random.normal([self.num_internal_nodes]), name='B'
    )
    if activation == 'sigmoid':
      self.activation = tf.keras.activations.sigmoid
    elif activation == 'smoother':
      self.activation = tfp.math.smootherstep
    elif activation == 'relu':
      self.activation = tf.keras.activations.relu
    elif activation == 'gelu':
      self.activation = tf.keras.activations.gelu
    # layer norm
    self.layer_norm = tf.keras.layers.LayerNormalization(axis=1)
    # dense layer
    self.dense = tf.keras.layers.Dense(num_classes)

  def __call__(self, inputs, training=False):
    hidden = self.activation(tf.matmul(inputs, self.weight) + self.bias)
    prediction = self.dense(hidden)
    return tf.nn.softmax(prediction, axis=-1)
