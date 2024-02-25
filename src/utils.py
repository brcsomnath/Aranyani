"""Commonly used Utilities."""

import math

import numpy as np
import tensorflow as tf


def construct_penalty_mask(tree_depth=4):
  num_internal_nodes = 2 ** tree_depth - 1
  mask = []
  for i in range(num_internal_nodes):
    power = math.floor(math.log2(i+1))
    factor = 1/(2** power)
    mask.append(factor)
  return np.array(mask).astype(np.float32)


def get_demographic_parity(y_predictions, y_protected):
  """Demographic parity.

  Args:
    y_predictions:
    y_protected:

  Returns:

  """
  predictions = np.array(y_predictions)
  protected_group = np.array(y_protected)

  protected_positive_rate = np.mean(predictions[protected_group == 1])
  unprotected_positive_rate = np.mean(predictions[protected_group == 0])

  raw_diff = protected_positive_rate - unprotected_positive_rate
  demographic_parity = np.abs(raw_diff)
  return demographic_parity, np.copysign(1, raw_diff)


def get_test_performance(model, x_test, y_test, a_test, data_dim=13):
  """Test performance.

  Args:
    model:
    x_test:
    y_test:
    a_test:
    data_dim:
  """
  accuracy = tf.keras.metrics.Accuracy()

  y_true = tf.convert_to_tensor(np.array(y_test))
  y_pred = tf.math.argmax(
      model(
          tf.convert_to_tensor(
              np.array(x_test, dtype=np.float32).reshape(-1, data_dim)
          )
      ),
      axis=-1,
  )

  accuracy.update_state(y_true, y_pred)
  print(f'Test Accuracy: {accuracy.result():.3f}')

  dp, _ = get_demographic_parity(y_pred, a_test)
  print(f'Test DP: {dp:.3f}')


def gauss_kernel(x1, x2, beta=1.0):
  assert len(x1.shape) == len(x2.shape)
  size = len(x1.shape)
  pairwise = tf.reduce_sum(
      (tf.expand_dims(x1, size - 1) - tf.expand_dims(x2, size - 2)) ** 2, size
  )
  return tf.exp(-0.5 * pairwise / beta)


def maximum_mean_discrepancy(x, y, kernel_scale=1.0):
  """Computes the Maximum Mean Discrepancy (MMD) between two batches of samples.

  Args:
      x: Tensor of shape [batch_size, feature_dim] representing the first batch
        of samples.
      y: Tensor of shape [batch_size, feature_dim] representing the second batch
        of samples.
      kernel_scale: Float specifying the scale of the kernel.

  Returns:
      mmd_loss: Scalar tensor representing the MMD loss.
  """

  # Compute the pairwise squared Euclidean distances
  k_xx = gauss_kernel(x, x, beta=kernel_scale)
  k_yy = gauss_kernel(y, y, beta=kernel_scale)
  k_xy = gauss_kernel(x, y, beta=kernel_scale)

  # Compute the MMD loss
  mmd_loss = (
      tf.reduce_mean(k_xx) - 2.0 * tf.reduce_mean(k_xy) + tf.reduce_mean(k_yy)
  )
  return mmd_loss