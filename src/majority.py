"""Script for online training."""

import random
import collections

import numpy as np
import tensorflow as tf
import wandb
import tqdm

import utils


def train_online(
    model,
    inputs,
    targets,
    protected_targets,
    batch_size=1,
    probability=0.5,
    local_run=False,
):
  """Online training function with node level fairness constraints.

  Args:
    model:
    inputs:
    targets:
    protected_targets:
    data_dim:
    batch_size:
    tree_depth:
    compute_fairness:
    lambda_const:
    num_trees:
    neutralize_gradients:
    base_gamma:
    constraint_type:
    gradient_type:
  
  Returns:
    DP:
    accuracies:
  """
  # with tf.device("GPU:0"): 
  dataset = tf.data.Dataset.from_tensor_slices(
      (inputs, targets, protected_targets)
  )
  dataset = dataset.batch(batch_size)
  majority_task_label = collections.Counter(targets).most_common(1)[0][0]

  optimizer = tf.keras.optimizers.Adam(learning_rate=2e-3)
  criteria = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

  avg_loss = tf.keras.metrics.Mean()
  avg_accuracy = tf.keras.metrics.Accuracy()
  avg_auc = tf.keras.metrics.AUC()

  y_predictions = []
  y_protected = []
  y_label = []
  demographic_parities = []
  accuracies = []


  iterations = tqdm.tqdm(dataset)
  for _, (
      inputs_batch,
      targets_batch,
      protected_batch) in enumerate(iterations):

    with tf.GradientTape(persistent=True) as tape:
      # predictions: [batch_size, num_class]
      # y: [num_trees, batch_size, num_internal_nodes]
      predictions, y = model(inputs_batch, training=True)

      # y_pred: [batch_size]
      y_pred = (
        tf.math.argmax(predictions, axis=-1)
        if random.random() < probability
        else tf.convert_to_tensor([majority_task_label] * inputs_batch.shape[0])
      )
      target_loss = criteria(y_true=targets_batch, y_pred=predictions)

      # update the task gradients w.r.t. current sample
      # d L_{CE}(y, \hat{y})/d\theta
      gradients = tape.gradient(target_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # get demographic parity scores
    y_label.extend(targets_batch.numpy())
    y_predictions.extend(y_pred.numpy())
    y_protected.extend(protected_batch.numpy())
    dp, _ = utils.get_demographic_parity(y_predictions, y_protected)

    # update the average accuracy
    avg_accuracy.update_state(targets_batch, y_pred)
    avg_auc.update_state(y_label, y_predictions)
    avg_loss.update_state(target_loss)

    demographic_parities.append(dp)
    accuracies.append(avg_accuracy.result())

    result = {"CE Loss": avg_loss.result(), 
              "Accuracy": avg_accuracy.result(),
              "AUC": avg_auc.result(),
              "DP": dp}
    iterations.set_description(
          f' CE Loss: {avg_loss.result():.3f},'
          f' Accuracy: { avg_accuracy.result():.3%},'
          f' AUC: {avg_auc.result():.3%},'
          f' DP: {dp:.8f},')
    
    if not local_run:
      wandb.log(result)
  return demographic_parities, accuracies
