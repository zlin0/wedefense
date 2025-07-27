#!/bin/bash

#SBATCH --job-name=resnet18 #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1
#SBATCH --partition=gpu   #queue
#SBATCH --mail-user="lzhan268@jh.edu"  #email for reporting
#SBATCH --mail-type=END,FAIL  #report types
#SBATCH --output=./logs/slurm-%A_%a.out
###SBATCH --array=0-2

#unset PYTHONPATH
#unset PYTHONHOME
#
source ~/.bashrc
conda activate wedefense_new
#conda activate /homes/kazi/isilnova/.conda/envs/wespeaker
#which python

# You may consider to submit them separately to desired machine.
./run.sh --stage 3 --stop_stage 3 >> logs/run.sh.stage3.log.$SLURM_JOB_ID 2>&1
./run.sh --stage 4 --stop_stage 6 >> logs/run.sh.stage4-6.log.$SLURM_JOB_ID 2>&1
./run.sh --stage 7 --stop_stage 7 >> logs/run.sh.stage7.log.$SLURM_JOB_ID 2>&1
