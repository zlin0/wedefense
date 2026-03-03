#!/bin/bash 

#SBATCH --job-name=asvspoof5 #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1
#SBATCH --partition=gpu   #queue
#SBATCH --partition=gpu-a100   # the a100 partition, for other gpu, use partition=gpu
#SBATCH --account=a100acct #we need this only for the a100 gpus
#SBATCH --mail-user="lzhan268@jh.edu"  #email for reporting
#SBATCH --mail-type=END,FAIL  #report types
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --array=4-4

#unset PYTHONPATH
#unset PYTHONHOME
#
source ~/.bashrc
conda activate wedefense_pip
#conda activate /homes/kazi/isilnova/.conda/envs/wespeaker
#which python

echo "[$(date)] Starting Job ID: $SLURM_JOB_ID" > logs/debug_$SLURM_JOB_ID.txt

CONFIGs=(singlereso-utt_gmlp_wavlm-large_frozen
	 singlereso-utt_gmlp_xlsr53_ft
	 singlereso-utt_gmlp_xlsr53_frozen
	 singlereso-utt_gmlp_wav2vec2-large_ft
	 singlereso-utt_gmlp_wav2vec2-large_frozen)
config_name=${CONFIGs[$SLURM_ARRAY_TASK_ID]} 


./run.sh --stage 2 --stop_stage 7 \
    --config conf/${config_name}.yaml \
    --exp_dir exp/${config_name}

echo "[$(date)] Finished Job ID: $SLURM_JOB_ID" >> logs/debug_$SLURM_JOB_ID.txt

