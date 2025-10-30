export MPICH_GPU_SUPPORT_ENABLED=1
# export NCCL_SOCKET_IFNAME=hsn
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export CXI_FORK_SAFE=1 
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1
export MIOPEN_FIND_MODE=1 #2
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

export FI_CXI_DEFAULT_CQ_SIZE=131072
export OMP_NUM_THREADS=8

export ROCM_PATH=/opt/rocm
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/lib:/opt/cray/libfabric/1.15.2.0/lib64

rm -rf ~/.cache/miopen