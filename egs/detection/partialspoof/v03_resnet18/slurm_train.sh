#!/bin/bash 

#SBATCH --job-name=data_prep #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --partition=cpu   #queue
#SBATCH --mail-user="lzhan268@jh.edu"  #email for reporting
#SBATCH --mail-type=END,FAIL  #report types
#SBATCH --output=./logs/slurm-%A_%a.out
###SBATCH --array=0-2

#unset PYTHONPATH
#unset PYTHONHOME
#
#source ~/.bashrc
#conda activate /homes/kazi/isilnova/.conda/envs/wespeaker
#which python

./run.sh --stage 2 --stop_stage 2 > logs/run.sh.stage2.log.1 2>&1
