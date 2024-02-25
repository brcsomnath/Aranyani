from absl import app
from train import *


def main(_):
  _, _ = train(
      lambda_const=FLAGS.lambda_const,
      dataset=FLAGS.dataset,
      mode=FLAGS.mode,
      model_type=FLAGS.model_type,
      max_iter=FLAGS.max_iter,
      depth=FLAGS.depth,
      num_trees=FLAGS.num_trees,
      compute_fairness=FLAGS.compute_fairness and FLAGS.lambda_const > 0,
      batch_size=FLAGS.batch_size,
      activation=FLAGS.activation,
      compute_mode=FLAGS.compute_mode,
      base_gamma=FLAGS.base_gamma,
      constraint_type=FLAGS.constraint_type,
      gradient_type=FLAGS.gradient_type,
      probability=FLAGS.probability,
      encoder_model=FLAGS.encoder_model,
      reservoir_size=FLAGS.reservoir_size,
      offline_loss_type=FLAGS.offline_loss_type,
      local_run=True,
  )

if __name__ == '__main__':
  app.run(main)