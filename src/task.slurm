#!/bin/bash

#SBATCH --job-name=lxl
#SBATCH --partition=dgx2
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

cd ~/lxl/ML-fracture/src
module load pytorch/1.6.0
pip install -r ../requirement.txt

bash run_bayes.sh
