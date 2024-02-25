import fdt
import forest
import tensorflow as tf

import numpy as np

def find_ancestor_indices(leaf_index):
  ancestor_indices = []
  while leaf_index > 1:
    flag = leaf_index % 2 == 0
    leaf_index = leaf_index // 2  # Integer division
    if flag:
      ancestor_indices.append(leaf_index)
    else:
      ancestor_indices.append(-leaf_index)
  return ancestor_indices

def test_leaf_probabilities():
  batch_size = 10
  data_dim = 100
  tree_depth = 2
  num_classes = 2
  num_iters = 100

  # test correctness for 100 samples
  # with different initializations
  for _ in range(num_iters):
    model = fdt.FairDecisionTree(
        data_dim=data_dim,
        tree_depth=tree_depth,
        num_classes=num_classes,
        activation="sigmoid",
    )
    x = tf.random.uniform([batch_size, data_dim], dtype=tf.float32)
    prediction, node_decisions, leaf_probs = model(x, training=True)

    assert prediction.shape == (batch_size, num_classes)
    assert node_decisions.shape == (batch_size, 2**tree_depth - 1)
    assert leaf_probs.shape == (batch_size, 2**tree_depth)

    for i in range(batch_size):
      leaf_prob = leaf_probs[i].numpy()
      node_decision = node_decisions[i].numpy()

      for idx, prob in enumerate(leaf_prob):
        ancestor_indices = find_ancestor_indices(idx + 2**tree_depth)

        p = 1
        for i in ancestor_indices:
          p *= (
              node_decision[i - 1]
              if i > 0
              else (1 - node_decision[abs(i) - 1])
          )
        # check correctness of leaf probabilities
        assert np.allclose(p, prob)

if __name__ == '__main__':
  test_leaf_probabilities()