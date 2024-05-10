#!/bin/bash

# charges to Daniel Khashabi's GPU account
#SBATCH--account=danielk_gpu
#SBATCH --partition=ica100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=36G
#SBATCH --time=4:00:00
#SBATCH --job-name="ssm-project-prompt-recovery-baseline"
# the file where your output will be printed
# writing %J will replace it with the job_id during runtime
#SBATCH --output=slurm/outputs/%J.log

# if you want to be notified when your job starts and ends, use this command
#SBATCH --mail-type=all 

# if you are using the email notification, this is your email (it needs to be @jhu.edu)
#SBATCH --mail-user=kkotkar1@jhu.edu
# save output in slurm_outputs directory

# module unload python
module load anaconda
# eval "$(conda shell.bash hook)"
conda activate nlpproj

# python import_dataset.py
python baselinellama.py