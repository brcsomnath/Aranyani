"""Fair classification decision tree."""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def construct_mask_matrix(tree_depth=3):
  """Construct mask matrix for efficient DT training."""

  num_internal_nodes = 2**tree_depth - 1
  num_leaves = 2**tree_depth

  mask_matrix = np.zeros([num_internal_nodes, num_leaves])

  # iterate for the leaves
  for idx, leaf_index in enumerate(
      range(2**tree_depth, 2 ** (tree_depth + 1))
  ):
    # iterate over ancestors
    ancestor = leaf_index
    last_ancestor = ancestor
    while ancestor > 1:
      ancestor = ancestor // 2
      mask_matrix[ancestor - 1, idx] = (
          1 if 2 * ancestor == last_ancestor else -1
      )
      last_ancestor = ancestor
  return mask_matrix


class FairDecisionTree(tf.Module):
  """Fair classification decision tree."""

  def __init__(self,
               data_dim,
               tree_depth,
               num_classes,
               activation='sigmoid',
               compute_mode='log'):
    """Constructor.

    Args: 
      data_dim: dimension of the data. 
      tree_depth: depth of the binary tree. 
      num_classes: number of target task classes.
      activation: activation function.
      compute_mode: log domain or normal.
    """

    super(FairDecisionTree, self).__init__()
    assert tree_depth > 1
    self.num_internal_nodes = 2**tree_depth - 1

    # internal node parameters
    self.weight = tf.Variable(
        tf.random.normal([data_dim, self.num_internal_nodes]), 
        name='W',
        trainable=True,
    )

    self.bias = tf.Variable(
        tf.random.normal([self.num_internal_nodes]),
        name='B',
        trainable=True,
    )

    if activation == 'sigmoid':
      self.activation = tf.keras.activations.sigmoid
    elif activation == 'smoother':
      self.activation = tfp.math.smootherstep
    elif activation == 'relu':
      self.activation = tf.keras.activations.relu
    elif activation == 'gelu':
      self.activation = tf.keras.activations.gelu

    # leaf parameters
    self.num_leaves = 2**tree_depth
    self.theta = tf.Variable(
        tf.random.uniform([self.num_leaves, num_classes]), 
        name='theta',
        trainable=True,
    )

    # mask parameters
    mask_matrix = construct_mask_matrix(tree_depth=tree_depth)
    self.mask_matrix = tf.constant(mask_matrix, 
                                   dtype=tf.float32,
                                   name='mask_matrix')
    self.ones_nodes = tf.constant(
        tf.nn.relu(-self.mask_matrix), 
        dtype=tf.float32,
        name='ones'
    )
    self.mask = tf.constant(
        tf.ones_like(self.mask_matrix) - tf.math.abs(self.mask_matrix),
        dtype=tf.float32,
        name='mask'
    )
    self.compute_mode = compute_mode

  def __call__(self, inputs, training=False, pred_type='categorical'):
    raw_node_decisions = self.activation(
        tf.matmul(inputs, self.weight) + self.bias)

    y = tf.expand_dims(raw_node_decisions, axis=2)
    y_repeated = tf.repeat(y, self.num_leaves, axis=2)
    z = tf.multiply(y_repeated, self.mask_matrix)

    # P \in [batch_size, num_internal_nodes, num_leaves]
    probs = tf.nn.relu(z) + (self.ones_nodes - tf.nn.relu(-z)) + self.mask

    # add numerical stability
    probs += 1e-8

    if self.compute_mode == 'log':
      probs = tf.math.log(probs)

      # axis=1 because it corresponds to internal nodes
      leaf_probs = tf.math.reduce_sum(probs, axis=1)
      theta = tf.expand_dims(self.theta, axis=0)

      prediction = tf.math.exp(tf.expand_dims(leaf_probs, axis=2) + theta)
      prediction = tf.reduce_sum(prediction, axis=1)

      leaf_probs = tf.math.exp(leaf_probs)
    else:
      leaf_probs = tf.math.reduce_prod(probs, axis=1)
      if pred_type == 'categorical':
          prediction = tf.matmul(leaf_probs, tf.nn.softmax(self.theta, axis=-1))
      else:
          prediction = tf.matmul(leaf_probs, self.theta)

    if training:
      return prediction, raw_node_decisions, leaf_probs
    return tf.nn.softmax(prediction, axis=-1)
