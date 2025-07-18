#!/bin/bash
#
#$ -cwd
#$ -V
#$ -N train_embd
#$ -o train_embd.out
#$ -e train_embd.err
#$ -l gpu=1,ram_free=10G,mem_free=10G,core=2,matylda4=2,scratch=0.5,gpu_ram=40G
####$ -q long.q@supergpu10,long.q@supergpu11,long.q@supergpu12,long.q@supergpu13,long.q@supergpu14,long.q@supergpu15,long.q@supergpu16,long.q@supergpu17,long.q@supergpu18

MAIN_DIR=/mnt/matylda4/qzhang/workspace/wedefense/egs/detection/partialspoof
cd ${MAIN_DIR}
unset PYTHONPATH
unset PYTHONHOME

source ~/.bashrc
conda activate /homes/kazi/isilnova/.conda/envs/wespeaker
which python

./run.sh --stage 2 --stop_stage 2 > logs/run.sh.stage2.log.1 2>&1

