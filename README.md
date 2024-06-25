# W&B Tutorial

This repository contains the materials used for Okubo Lab W&B tutorial held on 2024/06/24 hosted by Yue Chen and Xin Zheng.

## Requirements
A computer with the following packages installed. Any version should work!
- Python
- PyTorch
- torchvision
- wandb

While GPU allows you to train faster, we've designed the exercises so that it could be run on CPU.

## Files
- Two notebooks that contain exercises
  - `wandb_tutorial.ipynb`: basic work flow of using W&B
  - `wandb_sweeps_tutorial.ipynb`: how to perform hyperparameter optimization using W&B Sweeps   
- `utils.py` functions that are used in the tutorial (e.g. PyTorch model training)
- `data/MNIST/raw`: handwritten digit dataset (MNIST) used in the tutorial
- `wandb_tutorial.yml`: packages and versions that was used to create the repository
