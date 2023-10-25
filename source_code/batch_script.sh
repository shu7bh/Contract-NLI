#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH -w gnode084

python run.py

