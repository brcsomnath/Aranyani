"""Fair classification decision tree."""

import tensorflow as tf

from tensorflow.keras import layers, models

import fdt


class FairCLIPDecisionForest(tf.Module):
  """Fair classification decision tree."""

  def __init__(self,
               num_trees,
               data_dim,
               tree_depth,
               num_classes,
               activation='sigmoid',
               compute_mode='default',):
    """Constructor.

    Args:
      num_trees: number of trees in the forest.
      data_dim: dimension of the data. 
      tree_depth: depth of the binary tree. 
      num_classes: number of target task classes.
      activation: activation function.
      compute_mode: log or default.
    """

    super(FairCLIPDecisionForest, self).__init__()
    assert tree_depth > 1
    assert num_trees >= 1
    
    self.base_model = models.Sequential()
    self.base_model.add(layers.Flatten())
    
    self.layers = []
    for _ in range(num_trees):
      self.layers.append(fdt.FairDecisionTree(
          data_dim, tree_depth, num_classes, activation, compute_mode))

  def __call__(self, inputs, training=False):
    all_predictions = []
    all_node_decisions = []

    inputs = tf.cast(inputs, tf.float32)
    embeddings = self.base_model(inputs)

    for layer in self.layers:
      prediction, node_decisions, _ = layer(embeddings, 
                                            training=training,
                                            pred_type='categorical')
      all_predictions.append(prediction)
      all_node_decisions.append(node_decisions)

    final_prediction = tf.reduce_mean(tf.stack(all_predictions), axis=0)
    all_node_decisions = tf.stack(all_node_decisions, axis=0)
    if training:
      return final_prediction, all_node_decisions
    return tf.nn.softmax(final_prediction, axis=-1)
