import collections
import copy
import numpy as np
import tensorflow as tf
import tqdm
import wandb
import utils


def train_online(
    model,
    inputs,
    targets,
    protected_targets,
    data_dim=13,
    batch_size=1,
    tree_depth=3,
    compute_fairness=True,
    lambda_const=0.3,
    num_trees=3,
    base_gamma=0.9,
    constraint_type='node',
    gradient_type='vanilla',
    loss_type='ce',
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
    loss_type:
  
  Returns:
    DP:
    accuracies:
  """
  
  dataset = tf.data.Dataset.from_tensor_slices(
      (inputs, targets, protected_targets)
  )
  print(set(targets), set(protected_targets))
  dataset = dataset.batch(batch_size)
  num_internal_nodes = 2**tree_depth - 1
  optimizer = tf.keras.optimizers.Adam(learning_rate=2e-3)
  criteria = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
  avg_accuracy = tf.keras.metrics.Accuracy()

  gradient_b0_a0 = np.zeros([num_internal_nodes])
  gradient_b0_a1 = np.zeros([num_internal_nodes])

  gradient_w0_a0 = np.zeros([data_dim, num_internal_nodes])
  gradient_w0_a1 = np.zeros([data_dim, num_internal_nodes])

  gradient_w1_a0 = np.zeros([num_internal_nodes, 2])
  gradient_w1_a1 = np.zeros([num_internal_nodes, 2])
    
  gradient_b1_a0 = np.zeros([2])
  gradient_b1_a1 = np.zeros([2])

  agg_y_a0 = 0
  agg_y_a1 = 0
    
  grad_norm = []

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
      predictions = model(inputs_batch, training=True)
      # y_pred: [batch_size]
      y_pred = tf.math.argmax(predictions, axis=-1)
      target_loss = criteria(y_true=targets_batch, y_pred=predictions)
      # get demographic parity scores
      y_predictions.extend(y_pred.numpy())
      dp, dp_sign = dp_function(
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

      # update the task gradients w.r.t. current sample
      # d L_{CE}(y, \hat{y})/d\theta
      gradients = tape.gradient(target_loss, model.trainable_variables)
      total_gradients = gradients
      # fairness gradient computation
      # sign(E_{x~1}[n_i] - E_{x~0}[n_i]) *
      # (E_{x~1}[dn_i/d\theta] - E_{x~0}[dn_i/d\theta])
      # where n_i is the decision of the i-th internal node
      if compute_fairness:
        # iterate over the current batch and aggregate the gradients
        for i, a_label in enumerate(protected_batch.numpy()):
          protected_class_count[a_label] += 1
          # update the aggregate score based on protected label: A.
          if a_label == 0:
            agg_y_a0 += y_pred
          elif a_label == 1:
            agg_y_a1 += y_pred
            
          fair_gradients = tape.gradient(predictions[i], model.trainable_variables)
          factor = 1 / protected_class_count[a_label]
        
          # vanilla averaging
          # g_t = g_{t-1}*(1-1/t) + dL
          # stores the index of the tree to be updated
          for (i, fair_grad) in enumerate(fair_gradients):
            if i == 0:
              if a_label == 0:
                gradient_b0_a0 = gradient_b0_a0 * (1 - factor) + fair_grad.numpy() * factor
              elif a_label == 1:
                gradient_b0_a1 = gradient_b0_a1 * (1 - factor) + fair_grad.numpy() * factor
              else:
                raise NotImplementedError("Not implemented for non-binary a labels.")
            elif i == 1:
              if a_label == 0:
                gradient_w0_a0 = gradient_w0_a0 * (1 - factor) + fair_grad.numpy() * factor
              elif a_label == 1:
                gradient_w0_a1 = gradient_w0_a1 * (1 - factor) + fair_grad.numpy() * factor
              else:
                raise NotImplementedError("Not implemented for non-binary a labels.")
            elif i == 2:
              if a_label == 0:
                gradient_w1_a0 = gradient_w1_a0 * (1 - factor) + fair_grad.numpy() * factor
              elif a_label == 1:
                gradient_w1_a1 = gradient_w1_a1 * (1 - factor) + fair_grad.numpy() * factor
              else:
                raise NotImplementedError("Not implemented for non-binary a labels.")
            elif i == 3:
              if a_label == 0:
                gradient_b1_a0 = gradient_b1_a0 * (1 - factor) + fair_grad.numpy() * factor
              elif a_label == 1:
                gradient_b1_a1 = gradient_b1_a1 * (1 - factor) + fair_grad.numpy() * factor
              else:
                raise NotImplementedError("Not implemented for non-binary a labels.")

            
        # safety check
        if protected_class_count[0] == 0 or protected_class_count[1] == 0:
          continue
        
        correction_factor_0, correction_factor_1 = 1.0, 1.0
        total_gradients = []
        idx_b = 0
        idx_w = 0
        # iterate over all task gradients and add the fairness term
        for idx, grad in enumerate(gradients):
          signs_y = dp_sign
          if idx == 0:
            grad_diff = tf.convert_to_tensor(gradient_b0_a1) - tf.convert_to_tensor(gradient_b0_a1)
            grad_f = lambda_const * tf.multiply(signs_y, tf.cast(grad_diff, tf.float32))
          elif idx == 1:
            grad_diff = tf.convert_to_tensor(gradient_w0_a1) - tf.convert_to_tensor(gradient_w0_a0)
            grad_f = lambda_const * tf.multiply(signs_y, tf.cast(grad_diff, tf.float32))
            grad_norm.append(tf.norm(grad_f))
          elif idx == 2:
            grad_diff = tf.convert_to_tensor(gradient_w1_a1) - tf.convert_to_tensor(gradient_w1_a0)
            grad_f = lambda_const * tf.multiply(signs_y, tf.cast(grad_diff, tf.float32))
          elif idx == 3:
            grad_diff = tf.convert_to_tensor(gradient_b1_a1) - tf.convert_to_tensor(gradient_b1_a0)
            grad_f = lambda_const * tf.multiply(signs_y, tf.cast(grad_diff, tf.float32))
          
          total_gradients.append(grad + grad_f)
        total_gradients = tuple(total_gradients)
      optimizer.apply_gradients(zip(total_gradients, model.trainable_variables))
  return demographic_parities, grad_norm
