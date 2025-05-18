#!/bin/bash
#
#$ -cwd
#$ -V
#$ -N WeDefense
#$ -o out/out_$JOB_ID_$TASK_ID.out
#$ -e out/error_$JOB_ID_$TASK_ID.err
#$ -q long.q@supergpu8,long.q@supergpu9,long.q@supergpu10,long.q@supergpu11,long.q@supergpu12,long.q@supergpu13,long.q@supergpu14,long.q@supergpu15,long.q@supergpu16,long.q@supergpu17,long.q@supergpu18
#$ -pe smp 1
#$ -l gpu_ram=20G,ram_free=10G,mem_free=10G,gpu=1
#$ -t 1-1

source ~/.bashrc
conda activate /mnt/matylda6/pengjy/python_new/miniconda/envs/wespeaker
cd /mnt/matylda6/pengjy/shared_model_weights/lin/wedefense/egs/detection/asvspoof5/v15_ssl_mhfa

SSLMODELs=(
        MHFA_hubert 
        MHFA_wav2vec2
        MHFA_wavlm
        MHFA_wavlmplus
        MHFA_data2vec
        MHFA_wavlm_large
        MHFA_wav2vec2_large
        MHFA_hubert_large
        MHFA_data2vec_large
        MHFA_wav2vec2_xlsr53
        )
upstream=${SSLMODELs[$SGE_TASK_ID-1]} 

# echo "[$(date)] Starting Job ID: $SLURM_JOB_ID" > logs/debug_$SLURM_JOB_ID.txt
echo "[$(date)] Starting Job ID: $JOB_ID" > logs/debug_${JOB_ID}_${SGE_TASK_ID}.txt

sleep $((30 * ($SGE_TASK_ID - 1)))

./run.sh --stage 3 --stop_stage 7 --config conf/${upstream}.yaml --exp_dir exp/${upstream}_frozen
# ./run_large.sh --stage 8 --stop_stage 8 --config conf/${upstream}.yaml --ft_config conf/${upstream}-FT.yaml --exp_dir exp/${upstream}_frozen

# echo "[$(date)] Finished Job ID: $SLURM_JOB_ID" >> logs/debug_$SLURM_JOB_ID.txt
echo "[$(date)] Finished Job ID: $JOB_ID" >> logs/debug_${JOB_ID}_${SGE_TASK_ID}.txt
