# Aranyani
[![License: MIT](https://img.shields.io/badge/License-MIT-green``.svg)](https://opensource.org/licenses/MIT)

We present the implementation of the ICLR 2024 Spotlight paper:

> [**Enhancing Group Fairness in Online Settings Using Oblique Decision Forests**](https://arxiv.org/pdf/2310.11401.pdf),<br/>
[Somnath Basu Roy Chowdhury](https://www.cs.unc.edu/~somnath/)<sup>1</sup>, [Nicholas Monath](https://people.cs.umass.edu/~nmonath/)<sup>2</sup>, [Ahmad Beirami](https://sites.google.com/view/beirami)<sup>3</sup>, [Rahul Kidambi](https://rahulkidambi.github.io/)<sup>3</sup>, [Avinava Dubey](https://scholar.google.co.in/citations?user=tBbUAfsAAAAJ&hl=en)<sup>3</sup>, [Amr Ahmed](https://scholar.google.co.in/citations?user=tBbUAfsAAAAJ&hl=en)<sup>3</sup>, and [Snigdha Chaturvedi](https://sites.google.com/site/snigdhac/)<sup>1</sup>. <br>
UNC Chapel Hill<sup>1</sup>,  Google Deepmind<sup>2</sup>, Google Research<sup>3</sup>


## Overview
Fairness, especially group fairness, is an important consideration in the context of machine learning systems. The most commonly adopted group fairness-enhancing techniques are in-processing methods that rely on a mixture of a fairness objective (e.g., demographic parity) and a task-specific objective (e.g., cross-entropy) during the training process. However, when data arrives in an online fashion – one instance at a time – optimizing such fairness objectives poses several challenges. In particular, group fairness objectives are defined using expectations of predictions across different demographic groups. In the online setting, where the algorithm has access to a single instance at a time, estimating the group fairness objective requires additional storage and significantly more computation (e.g., forward/backward passes) than the task-specific objective at every time step. In this paper, we propose <i>Aranyani</i>, an ensemble of oblique decision trees, to make fair decisions in online settings. The hierarchical tree structure of <i>Aranyani</i> enables parameter isolation and allows us to efficiently compute the fairness gradients using aggregate statistics of previous decisions, eliminating the need for additional storage and forward/backward passes. We also present an efficient framework to train <i>Aranyani</i> and theoretically analyze several of its properties. We conduct empirical evaluations on 5 publicly available benchmarks (including vision and language datasets) to show that <i>Aranyani</i> achieves a better accuracy-fairness trade-off compared to baseline approaches.

![alt text](https://github.com/brcsomnath/Aranyani/blob/main/files/oblique_tree.png?raw=true)

## Installation
The simplest way to run our implementation is to create with a new conda environment.
```
conda create -n arnya python=3.8
source activate arnya
pip install -r requirements.txt
```

## Running Aranyani

To run aranyani, use the following command:

```
cd src/
python main.py --dataset <name> --mode node --batch_size 1 --lambda_const 1.0
```

The dataset names used to run the above command is presented below:

```
{
  'Adult': 'adult',
  'Census': 'census',
  'COMPAS': 'compas',
  'CelebA': 'celeba',
  'CivilComments': 'jigsaw'
}
```

The exact data used in our experiments can be found [here](https://drive.google.com/file/d/1ibViykIbtfumtTVFTVaHwWUr_vS6VH4f/view?usp=sharing).

## Running Aranyani Using WandB

To get the exact accuracy-fairness tradeoffs reported in the paper you could use the exact configs and run a sweep over different hyperparameters using Weights & Biases. The wandb commands are shown below:

```
cd src/
wandb sweep -p <project_name> configs/<config_name>.yaml
```

After running the above command, wandb will generate a command with sweep id. You need to copy and paste that command. The output command will have the following format:

```
wandb agent <username>/<project_name>/<sweep_id>
```


## Reference


```
@inproceedings{chowdhury2024enhancing,
  title={Enhancing Group Fairness in Online Settings Using Oblique Decision Forests},
  author={Somnath Basu Roy Chowdhury and 
          Nicholas Monath and 
          Ahmad Beirami and 
          Rahul Kidambi and 
          Kumar Avinava Dubey and 
          Amr Ahmed and 
          Snigdha Chaturvedi},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=E1NxN5QMOE}
}
```