"""Script for online training."""

import collections
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
  num_internal_nodes = 2**tree_depth - 1
  optimizer = tf.keras.optimizers.Adam(learning_rate=2e-3)
  criteria = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
  avg_accuracy = tf.keras.metrics.Accuracy()
  gradient_w_a0 = np.zeros([num_trees, data_dim, num_internal_nodes])
  gradient_w_a1 = np.zeros([num_trees, data_dim, num_internal_nodes])
  gradient_b_a0 = np.zeros([num_trees, num_internal_nodes])
  gradient_b_a1 = np.zeros([num_trees, num_internal_nodes])
  agg_y_a0 = tf.zeros([num_trees, num_internal_nodes], name='agg_Y0')
  agg_y_a1 = tf.zeros([num_trees, num_internal_nodes], name='agg_Y1')

  # avoid runtime error
  protected_class_count = collections.defaultdict(int)
  dp_function = utils.get_demographic_parity
  avg_loss = tf.keras.metrics.Mean()
  avg_auc = tf.keras.metrics.AUC()
  y_predictions = []
  demographic_parities = []
  accuracies = []

  # hyperparameters
  huber_loss_delta = 0.01

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
            agg_y_a0 += node_decisions[:, i]
          elif a_label == 1:
            agg_y_a1 += node_decisions[:, i]
          if constraint_type == 'node':
            # compute the gradients w.r.t. node-level decisions
            fair_gradients = tape.gradient(node_decisions[:, i],
                                           model.layers.trainable_variables)
          elif constraint_type == 'leaf':
            # compute the gradients w.r.t. final forest decisions
            fair_gradients = tape.gradient(
                predictions[i], model.trainable_variables
            )
          else:
            raise ValueError('Constraint type not identified.')
            
          # stores the index of the tree to be updated
          idx_b = -1
          idx_w = -1
          for fair_grad in fair_gradients:
            if fair_grad is None:
              continue
            if len(fair_grad.shape) == 1:
              # selected parameter B: bias
              # select the group to be updated
              gradient_theta = gradient_b_a0 if a_label == 0 else gradient_b_a1
              idx_b += 1
              idx_theta = idx_b
            elif len(fair_grad.shape) == 2 and fair_grad.shape[0] == data_dim:
              # selected parameter W: weight
              # select the group to be updated
              gradient_theta = gradient_w_a0 if a_label == 0 else gradient_w_a1
              idx_w += 1
              idx_theta = idx_w
            else:
              # ignore fairness gradients for leaf parameters
              continue
            if gradient_type == 'momentum':
              # momentum term, g_t = \beta*g_{t-1} + dL
              gradient_theta[idx_theta] = (
                  gradient_theta[idx_theta] * base_gamma
                  + fair_grad.numpy()
              )
            elif gradient_type == 'ema':
              # momentum term, m_t = \beta*m_{t-1} + dL
              # g_t = (1 - beta)/(1-beta^t) * m_t
              # correction factor added later
              gradient_theta[idx_theta] = (
                  gradient_theta[idx_theta] * base_gamma
                  + fair_grad.numpy()
              )
            else:
              # vanilla averaging
              # g_t = g_{t-1}*(1-1/t) + dL
              factor = 1 / protected_class_count[a_label]
              gradient_theta[idx_theta] = (
                  gradient_theta[idx_theta] * (1 - factor)
                  + fair_grad.numpy() * factor
              )
        # safety check
        if protected_class_count[0] == 0 or protected_class_count[1] == 0:
          continue
        if gradient_type == 'ema':
          # enable correction factors
          correction_factor_0 = (1 - base_gamma) / (
              1 - base_gamma ** protected_class_count[0]
          )
          correction_factor_1 = (1 - base_gamma) / (
              1 - base_gamma ** protected_class_count[1]
          )
        else:
          correction_factor_0, correction_factor_1 = 1.0, 1.0
        total_gradients = []
        idx_b = 0
        idx_w = 0
        # iterate over all task gradients and add the fairness term
        for idx, grad in enumerate(gradients):
          tree_id = idx // 3 # 3 set of parameters per tree, (W, B, \bf \Theta)
          
          # compute sign(E_{x~1}[n_i] - E_{x~0}[n_i]) for tree_id-th tree
          F = agg_y_a1[tree_id] / protected_class_count[1]- agg_y_a0[tree_id] / protected_class_count[0]
          if constraint_type == 'node':
            signs_y = tf.math.sign(F - huber_loss_delta/2)
          else:
            signs_y = dp_sign
          
          if len(grad.shape) == 1:
            # bias parameter selection: B
            # compute: (E_{x~1}[dn_i/dB] - E_{x~0}[dn_i/dB])
            grad_b_diff = tf.convert_to_tensor(
                gradient_b_a1[idx_b] * correction_factor_1
            ) - tf.convert_to_tensor(
                gradient_b_a0[idx_b] * correction_factor_0)
            grad_b_diff = tf.cast(grad_b_diff, tf.float32)
            
            huber_check = tf.cast(tf.math.abs(F) < huber_loss_delta, tf.float32)
            huber_quadratic = tf.multiply(huber_check * F, grad_b_diff)
            huber_abs = tf.multiply(tf.multiply(1-huber_check, signs_y), grad_b_diff)

            grad_b = lambda_const * (huber_quadratic + huber_abs)
            total_gradients.append(grad + grad_b)
            idx_b = idx_b + 1
          elif len(grad.shape) == 2 and grad.shape[0] == data_dim:
            # weight parameter selection: W
            # compute: (E_{x~1}[dn_i/dW] - E_{x~0}[dn_i/dW])
            grad_w_diff = tf.convert_to_tensor(
                gradient_w_a1[idx_w] * correction_factor_1
            ) - tf.convert_to_tensor(
                gradient_w_a0[idx_w] * correction_factor_0)
            grad_w_diff = tf.cast(grad_w_diff, tf.float32)
            
            huber_check = tf.cast(tf.math.abs(F) < huber_loss_delta, tf.float32)
            huber_quadratic = tf.multiply(tf.multiply(huber_check, F), grad_w_diff)
            huber_abs = tf.multiply(tf.multiply(1-huber_check, signs_y), grad_w_diff)

            grad_w = lambda_const * (huber_quadratic + huber_abs)
            total_gradients.append(grad + grad_w)
            idx_w = idx_w + 1
          else:
            # no fairness gradients for the leaf parameters
            total_gradients.append(grad)
        total_gradients = tuple(total_gradients)
      optimizer.apply_gradients(zip(total_gradients, model.trainable_variables))
  return demographic_parities, accuracies
