"""Script for online training."""

import tqdm
import wandb
import utils
import collections
import numpy as np
import tensorflow as tf


def batch_data(*lists, batch_size):
  num_samples = len(lists[0]) 

  batches = []
  for i in range(0, num_samples, batch_size):
    batch = tuple([lst[i:i + batch_size] for lst in lists])
    batches.append(batch)
  return batches

def train_online(
    model,
    inputs,
    targets,
    protected_targets,
    batch_size=1,
    local_run=False,
    label_type='categorical',
):
  """Online training function with node level fairness constraints.
  Args:
    model:
    inputs:
    targets:
    protected_targets:
    batch_size:
  
  Returns:
    DP:
    accuracies:
  """

  row_sums = inputs.sum(axis=1)
  inputs = inputs / row_sums[:, np.newaxis]

  dataset = batch_data(inputs, targets, protected_targets, batch_size=batch_size)

  # avoid runtime error
  dp_function = utils.get_demographic_parity
  avg_loss = tf.keras.metrics.Mean()
  avg_auc = tf.keras.metrics.AUC()
  avg_score = tf.keras.metrics.Accuracy() if label_type == 'categorical' else tf.keras.metrics.MeanSquaredError()
  majority_task_label = collections.Counter(targets).most_common(1)[0][0]

  y_predictions = []
  demographic_parities = []
  accuracies = []

  iterations = tqdm.tqdm(dataset)
  for (inputs_batch, targets_batch, protected_batch) in iterations:


    try:
      y_pred = model.predict(inputs_batch)
      model.partial_fit(inputs_batch, targets_batch)
    except Exception as e:
      print(e)
      # if the AHT can't handle input, assume random output
      y_pred = [majority_task_label]
    
    y_predictions.extend(y_pred)

    dp, _ = dp_function(
          y_predictions, protected_targets[: len(y_predictions)])
    
    # update the average accuracy
    avg_score.update_state(targets_batch, y_pred)
    avg_auc.update_state(targets_batch, y_pred)
    
    iterations.set_description(
        f' Accuracy: { avg_score.result():.3%},'
        f' AUC: {avg_auc.result():.3%},'
        f' DP: {dp:.5f},'
    )
    results = {
        'Accuracy': avg_score.result(),
        'AUC': avg_auc.result(),
        'DP': dp,
    }

    if not local_run:
      wandb.log(results)
  return demographic_parities, accuracies