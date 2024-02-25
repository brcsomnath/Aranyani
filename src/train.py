"""Driver code."""


import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from absl import flags

import data
import mlp

import forest
import majority
import mlp_trainer
import aranyani

import reservoir
import hoeffding_tree

from skmultiflow.trees import HoeffdingTree
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier

import utils
import clip_forest


flags.DEFINE_string('sweep_id', '-1', 'Wandb sweep ID.')
flags.DEFINE_float('lambda_const', 0.1, 'Cell to run.')
flags.DEFINE_string('dataset', 'civil', 'Dataset name.')
flags.DEFINE_string('mode', 'node', 'Loss mode.')
flags.DEFINE_integer('max_iter', 1, 'Total number of iterations.')
flags.DEFINE_integer('depth', 4, 'Tree depth.')
flags.DEFINE_integer('num_trees', 3, 'Number of trees.')
flags.DEFINE_bool(
    'compute_fairness', True, 'Whether to apply fairness constraints.'
)
flags.DEFINE_integer('batch_size', 1, 'Samples in an online batch.')
flags.DEFINE_string('activation', 'sigmoid', 'Activation function.')
flags.DEFINE_string('model_type', 'forest', 'Type of f(x).')
flags.DEFINE_string('compute_mode', 'default', 'log or default.')
flags.DEFINE_string('base_gamma', None, 'gamma for the gradients.')
flags.DEFINE_string('constraint_type', 'node', '[node, leaf]')
flags.DEFINE_string('gradient_type', 'vanilla', '[vanilla, momentum, ema]')
flags.DEFINE_float('probability', 0.5,
                   'Probability of selection in majority baseline.')
flags.DEFINE_string('encoder_model', 'instructor', '[bert, instructor]')
flags.DEFINE_integer('reservoir_size', 100, 'size of reservoir.')
flags.DEFINE_string('offline_loss_type', 'mmd', '[mmd, l2, l1]')

FLAGS = flags.FLAGS

def train(
    mode='node',
    dataset='civil',
    lambda_const=1,
    model_type='forest',
    max_iter=3,
    depth=4,
    num_trees=3,
    compute_fairness=True,
    batch_size=1,
    activation='sigmoid',
    compute_mode='default',
    base_gamma=None,
    constraint_type='node',
    gradient_type='vanilla',
    probability=0.5,
    encoder_model='instructor',
    reservoir_size=100,
    offline_loss_type='mmd',
    local_run=False,
):
  """Training function.

  Args:
    mode:
    dataset:
    lambda_const:
    model_type:
    max_iter:
    depth:
    num_trees:
    compute_fairness:
    batch_size:
    activation:
    compute_mode:
    base_gamma:
    constraint_type:
    gradient_type:
    probability:

  Returns:

  """

  all_dps = []
  all_accuracies = []

  data_dim, num_class = None, None
  if dataset == 'adult':
    data_dim = 14
    num_class = 2
    x_train, _, y_train, _, a_train, _ = data.read_adult()
  elif dataset == 'census':
    data_dim = 40
    num_class = 2
    x_train, _, y_train, _, a_train, _ = data.read_census()
  elif dataset == 'compas':
    data_dim = 10
    x_train, y_train, a_train = data.read_compas()
    num_class = 2
  elif dataset == 'jigsaw':
    data_dim = 768
    x_train, y_train, a_train = data.read_jigsaw()
    num_class = 2
  elif dataset == 'celeba':
    data_dim = 768
    x_train, y_train, a_train = data.read_celeba()
    num_class = 2
  else:
    x_train, y_train, a_train = [], [], []

  base_dp, _ = utils.get_demographic_parity(y_train, a_train)
  print(f'DP in the original dataset: {base_dp}')

  for _ in range(max_iter):
    model = None
    if model_type == 'mlp':
      model = mlp.FairReLUNetwork(
          data_dim=data_dim,
          tree_depth=depth,
          num_classes=num_class,
          activation=activation,
      )
    elif model_type == 'ht':
      model = HoeffdingTree()

    elif model_type == 'aht':
      model = HoeffdingAdaptiveTreeClassifier()

    elif model_type == 'forest':
      model = forest.FairDecisionForest(
          num_trees=num_trees,
          data_dim=data_dim,
          tree_depth=depth,
          num_classes=num_class,
          activation=activation,
          compute_mode=compute_mode,
      )
      if dataset in ['celeba']:
          model = clip_forest.FairCLIPDecisionForest(
              num_trees=num_trees,
              data_dim=data_dim,
              tree_depth=depth,
              num_classes=num_class,
              activation=activation,
              compute_mode=compute_mode,
          )

    if mode == 'node' and model_type == 'forest':
      train_func = aranyani.train_online
      
      dp, accuracies = train_func(
          model,
          x_train,
          y_train,
          a_train,
          data_dim=data_dim,
          batch_size=batch_size,
          tree_depth=depth,
          compute_fairness=compute_fairness,
          lambda_const=lambda_const,
          num_trees=num_trees,
          base_gamma=base_gamma,
          constraint_type=constraint_type,
          gradient_type=gradient_type,
          local_run=local_run,
      )
    elif mode == 'majority':
        train_func = majority.train_online
        dp, accuracies = train_func(
            model,
            x_train,
            y_train,
            a_train,
            batch_size=batch_size,
            probability=probability,
            local_run=local_run,
        )
    elif model_type == 'mlp':
        dp, accuracies = mlp_trainer.train_online(
          model,
          x_train,
          y_train,
          a_train,
          data_dim=data_dim,
          batch_size=batch_size,
          tree_depth=depth,
          compute_fairness=compute_fairness,
          lambda_const=lambda_const,
          num_trees=num_trees,
          base_gamma=base_gamma,
          constraint_type=constraint_type,
          gradient_type=gradient_type,
          local_run=local_run
      )
    elif model_type in ['ht', 'aht']:
      dp, accuracies = hoeffding_tree.train_online(
        model,
        x_train,
        y_train,
        a_train,
        batch_size=batch_size,
        local_run=local_run,
        label_type = 'categorical',
      )
    elif mode == 'reservoir':
      dp, accuracies = reservoir.train_online(
        model,
        x_train,
        y_train,
        a_train,
        batch_size=batch_size,
        tree_depth=depth,
        compute_fairness=compute_fairness,
        lambda_const=lambda_const,
        reservoir_size=reservoir_size,
        local_run=local_run,)
    else:
      dp, accuracies = [], []

    all_dps.append(dp)
    all_accuracies.append(accuracies)
    del model
  return all_dps, all_accuracies
