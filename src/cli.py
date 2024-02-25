import yaml

from train import *

def main(_):
  """Main function."""
  with open('configs/adult_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

  run = wandb.init(
    config=config,
    settings=wandb.Settings(program_relpath="main.py", 
    disable_git=True, 
    save_code=False))
  config = wandb.config
  run.tags += (f'dataset-{config.dataset}',
      f'lambda-{config.lambda_const}',
      f'batch_size-{config.batch_size}',
      f'gradient_type-{config.gradient_type}')

  _, _ = train(
      lambda_const=config.lambda_const,
      dataset=config.dataset,
      mode=config.mode,
      model_type=config.model_type,
      max_iter=config.max_iter,
      depth=config.depth,
      num_trees=config.num_trees,
      compute_fairness=config.compute_fairness and config.lambda_const > 0,
      batch_size=config.batch_size,
      activation=config.activation,
      compute_mode=config.compute_mode,
      base_gamma=config.base_gamma,
      constraint_type=config.constraint_type,
      gradient_type=config.gradient_type,
      probability=config.probability,
      encoder_model=config.encoder_model,
      reservoir_size=config.reservoir_size,
      offline_loss_type=config.offline_loss_type,
  )
  run.finish()

if __name__ == '__main__':
  app.run(main)