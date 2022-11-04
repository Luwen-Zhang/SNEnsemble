#!/bin/bash

#SBATCH --job-name=lxl
#SBATCH --partition=small
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load miniconda3
module load cuda/10.2
source activate pytorch-env
cd ~/lxl/ML-fracture/src

bash run.sh
