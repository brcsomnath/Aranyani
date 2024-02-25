"""Script for online training."""

import random
import collections
import tensorflow as tf
import tqdm
import wandb
import utils


def train_online(
    model,
    inputs,
    targets,
    protected_targets,
    batch_size=1,
    tree_depth=3,
    compute_fairness=True,
    lambda_const=0.3,
    reservoir_size=100,
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
  
  dataset = tf.data.Dataset.from_tensor_slices(
      (inputs, targets, protected_targets)
  )
  dataset = dataset.batch(batch_size)
  optimizer = tf.keras.optimizers.Adam(learning_rate=2e-3)
  criteria = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
  avg_accuracy = tf.keras.metrics.Accuracy()
  reservoir_0 = []
  reservoir_1 = []

  # avoid runtime error
  protected_class_count = collections.defaultdict(int)
  dp_function = utils.get_demographic_parity
  avg_loss = tf.keras.metrics.Mean()
  avg_auc = tf.keras.metrics.AUC()
  y_predictions = []
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
      predictions, node_decisions = model(inputs_batch, training=True)
      # y_pred: [batch_size]
      y_pred = tf.math.argmax(predictions, axis=-1)
      target_loss = criteria(y_true=targets_batch, y_pred=predictions)
      # get demographic parity scores
      y_predictions.extend(y_pred.numpy())
      dp, _ = dp_function(
          y_predictions, protected_targets[: len(y_predictions)])
      # update the average accuracy
      avg_accuracy.update_state(targets_batch, y_pred)
      avg_auc.update_state(targets_batch, y_pred)
      avg_loss.update_state(target_loss)
      iterations.set_description(
          f' CE Loss: {avg_loss.result():.3f},'
          f' Accuracy: { avg_accuracy.result():.3%},'
          f' AUC: {avg_auc.result():.3%},'
          f' DP: {dp:.5f},'
      )
      results = {
          'CE loss': avg_loss.result(),
          'Accuracy': avg_accuracy.result(),
          'AUC': avg_auc.result(),
          'DP': dp,
      }

      if not local_run:
        wandb.log(results)

      
      if compute_fairness:
        for i, a_label in enumerate(protected_batch.numpy()):
          protected_class_count[a_label] += 1

          if a_label == 0:
            if len(reservoir_0) < reservoir_size:
              reservoir_0.append(inputs_batch[i].numpy())
            else:
              j = random.randrange(protected_class_count[a_label])
              if j < reservoir_size:
                reservoir_0[j] = inputs_batch[i]
          else:
            if len(reservoir_1) < reservoir_size:
              reservoir_1.append(inputs_batch[i].numpy())
            else:
              j = random.randrange(protected_class_count[a_label])
              if j < reservoir_size:
                reservoir_1[j] = inputs_batch[i]          

        if len(reservoir_0) < 2 or len(reservoir_1) < 2:
          gradients = tape.gradient(target_loss, model.trainable_variables)
          optimizer.apply_gradients(zip(gradients, model.trainable_variables))
          continue

        inputs_0 = tf.convert_to_tensor([x for x in reservoir_0])
        inputs_1 = tf.convert_to_tensor([x for x in reservoir_1])

        outputs_0 = tf.math.reduce_mean(model(inputs_0, training=True)[1], axis=1)
        outputs_1 = tf.math.reduce_mean(model(inputs_1, training=True)[1], axis=1)

        dp_loss = tf.math.abs(outputs_1 - outputs_0)

        gradients = tape.gradient(target_loss + lambda_const * dp_loss, 
                                  model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      else:
        gradients = tape.gradient(target_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return demographic_parities, accuracies
