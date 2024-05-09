#!/bin/bash

#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:0
#SBATCH --job-name="ssm-project-prompt-recovery-baseline"
# save output in slurm_outputs directory
#SBATCH --output=slurm_outputs/slurm-%j.out
#SBATCH --mem=16G

module load anaconda
conda activate ssm_project

# python import_dataset.py
python baselinellama.py