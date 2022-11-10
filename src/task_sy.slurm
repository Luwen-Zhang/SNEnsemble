#!/bin/bash

#SBATCH --job-name=lxl
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

module load miniconda3
module load cuda/11.5.0
source activate pytorch-env
cd ~/lxl/ML-fracture/src

bash run.sh
