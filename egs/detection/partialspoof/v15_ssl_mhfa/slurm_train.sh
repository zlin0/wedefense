#!/bin/bash 

#SBATCH --job-name=train_mhfa #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1
#SBATCH --partition=gpu   #queue
#SBATCH --mail-user="lzhan268@jh.edu"  #email for reporting
#SBATCH --mail-type=END,FAIL  #report types
#SBATCH --output=./logs/slurm-%j.out
###SBATCH --array=0-2

#unset PYTHONPATH
#unset PYTHONHOME
#
source ~/.bashrc
conda activate wedefense_fair
#conda activate /homes/kazi/isilnova/.conda/envs/wespeaker
#which python

echo "[$(date)] Starting Job ID: $SLURM_JOB_ID" > logs/debug_$SLURM_JOB_ID.txt

./run.sh --stage 3 --stop_stage 3

echo "[$(date)] Finished Job ID: $SLURM_JOB_ID" >> logs/debug_$SLURM_JOB_ID.txt

