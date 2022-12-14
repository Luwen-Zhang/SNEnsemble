#!/bin/bash

#SBATCH --job-name=lxl
#SBATCH --partition=small
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20

module load miniconda3
module load cuda/10.2
source activate pytorch-env
cd ~/lxl/ML-fracture

bash run.sh |& tee out.txt
