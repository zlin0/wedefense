#!/bin/bash
set -ex

# This script prepares the MUSAN and RIR_NOISES datasets.
# It creates wav.scp files and generates LMDB databases.

# NOTE: Please make sure these paths are correct for your environment.
# These directories are expected to be created by local/prepare_data_musan_rir.sh
MUSAN_dir=./data/raw_data/musan
RIRS_dir=./data/raw_data/RIRS_NOISES


if [ ! -e data/musan/wav.scp ] && [ ! -e data/rirs/wav.scp ]; then
    bash ./local/prepare_data_musan_rir.sh --stage 1 --stop-stage 3
    # This will download musan and rir, and create:
    #  data/download_data/{musan.tar.gz, rirs_noises.zip},
    #  data/raw_data/{RIRS_NOISES, musan} and
    #  data/{musan, rirs}/wav.scp
fi

if [ ! -e data/musan/lmdb/lock.mdb ] && [ ! -e data/rirs/lmdb/lock.mdb ]; then
    # generate lmdb for musan and rir.
    # NOTE: LMDB may not work on NFS. We create it locally and then rsync.
    if [[ $(hostname -A) == *clsp.jhu.edu* ]]; then
    echo "test"
            (
            python tools/make_lmdb.py data/musan/wav.scp "${HOME}/local_lmdb/musan/lmdb" && \
            rsync -av "${HOME}/local_lmdb/musan/lmdb" data/musan/lmdb
            ) &
            (
            python tools/make_lmdb.py data/rirs/wav.scp "${HOME}/local_lmdb/rirs/lmdb" && \
            rsync -av "${HOME}/local_lmdb/rirs/lmdb" data/rirs/lmdb
            ) &
            wait
    else
        python tools/make_lmdb.py data/musan/wav.scp data/musan/lmdb &
        python tools/make_lmdb.py data/rirs/wav.scp data/rirs/lmdb &
        wait
    fi
    echo "Complete processing musan and rirs."
else
    echo "We Already have musan and rirs."
fi