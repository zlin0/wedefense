#!/bin/bash

time=$(date +"%Y-%m-%d")
echo "Date: $time"

WORK_DIR=/scratch/project_465001402/junyi/sv_anti_dev/wedefense/egs/detection/asvspoof5/v15_ssl_mhfa_pruning
cd $WORK_DIR

my_folder="${WORK_DIR}/log/${time}/"

mkdir -p "${my_folder}"

sbatch -J "WeDfe_Pruning_v0" \
  --time "24:00:00" \
  --array "1-1" \
  -o "${my_folder}/output_%x_%j_%a.txt" \
  -e "${my_folder}/error_%x_%j_%a.txt" \
    slurm_train.sh
