#!/bin/bash

# Check GPU status
if [ $SLURM_LOCALID -eq 0 ]; then
    rocm-smi
fi
sleep 2

# Set MIOpen cache path
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
if [ ! -d ~/.cache/miopen ]; then
    mkdir -p ~/.cache/miopen
fi

# Print task binding and GPU information
taskset -pc 0-$((SLURM_CPUS_ON_NODE-1)) $$
echo "Rank $SLURM_PROCID --> $(taskset -p $$); GPU $ROCR_VISIBLE_DEVICES"

# PyTorch distributed configuration
export MASTER_PORT=$((RANDOM + 20000))
export WORLD_SIZE=$SLURM_NPROCS
export RANK=$SLURM_PROCID

# NCCL configuration
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_SOCKET_IFNAME=$(ls /sys/class/net | grep 'hsn' | paste -sd ',')
export NCCL_NET_GDR_LEVEL=3
export NCCL_DEBUG=INFO

# PyTorch HIP configuration
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

# OpenMP configuration
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-8}

# ROCm configuration
export ROCM_PATH=/opt/rocm

# Disable MIOpen algorithm search to reduce log output
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_LOG_LEVEL=0
export MIOPEN_DEBUG_DISABLE_FIND_DB=1

# Dynamic library paths
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/lib:/opt/cray/libfabric/1.15.2.0/lib64

export WANDB_MODE=offline

# Automatically clean up cache
# trap "rm -rf /tmp/$(whoami)-miopen-cache-$SLURM_NODEID" EXIT