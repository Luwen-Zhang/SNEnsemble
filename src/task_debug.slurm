#!/bin/bash

#SBATCH --job-name=lxl
#SBATCH --partition=debuga100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load miniconda3
module load cuda/11.5.0
source activate pytorch-env
cd ~/lxl/ML-fracture/src

bash run.sh
