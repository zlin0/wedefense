#!/bin/bash

#SBATCH --job-name=inf_loc_mhfa-fr #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1
#SBATCH --partition=gpu   #queue
##SBATCH --partition=gpu-a100   #queue
##SBATCH --account=a100acct
#SBATCH --mail-user="lzhan268@jh.edu"  #email for reporting
#SBATCH --mail-type=END,FAIL  #report types
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --array=1

#unset PYTHONPATH
#unset PYTHONHOME
#
echo "[$(date)] Starting Job ID: $SLURM_JOB_ID"

if [ ! -d logs ]; then
    mkdir logs
fi

source ~/.bashrc
conda activate wedefense_new
which python
#CONFIGs=(resnet
#	resnet_earlystop
#	resnet_lfcc_torchaudio)
#
DSETs=(dev test)
dset=${DSETs[$SLURM_ARRAY_TASK_ID]}

#CONFIGs=
#config_name=${CONFIGs[$SLURM_ARRAY_TASK_ID]}
#config_name=MHFA_wav2vec2_xlsr53-FT-1stage
config_name=MHFA_wav2vec2_xlsr53-Frozen
#config_name=MHFA_wav2vec2_xlsr53-Frozen-bs128
#./run.sh --stage 2 --stop_stage 2 --data_type raw

# You may consider to submit them separately to desired machine.
./run.sh --stage 4 --stop_stage 7 \
    --DSETs ${dset} \
    --config conf/${config_name}.yaml \
    --exp_dir exp/${config_name}


echo "[$(date)] Finished Job ID: $SLURM_JOB_ID"
