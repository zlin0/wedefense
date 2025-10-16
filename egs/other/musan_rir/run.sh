#!/bin/bash
set -ex


# bash ./local/prepare_data_musan_rir.sh --stage$ 1 --stop-stage 3
# This will download musan and rir, and create:
#  data/download_data/{musan.tar.gz, rirs_noises.zip},
#  data/raw_data/{RIRS_NOISES, musan} and
#  data/{musan, rirs}/wav.scp
if [ !-e data/musan/wav.scp ] && [ !-e data/rirs/wav.scp ]; then
    # Create wav.scp for MUSAN and RIRS
    find "${MUSAN_dir}" -name "*.wav" | awk -F"/" '{print $NF,$0}' | sort > data/musan/wav.scp
    find "${RIRS_dir}" -name "*.wav" | awk -F"/" '{print $NF,$0}' | sort > data/rirs/wav.scp
fi

if [ !-e data/musan/lmdb ] && [ !-e data/rirs/lmdb ]; then
    # generate lmdb for musan and rir.
    # NOTE: LMDB may not work on NFS. We create it locally and then rsync.
    if [[ $(hostname -f) == *clsp.jhu.edu   ]]; then
            (
            python tools/make_lmdb.py data/musan/wav.scp "${HOME}/local_lmdb/musan/lmdb" && \
            rsync -av "${HOME}/local_lmdb/musan/lmdb" data/musan/lmdb
            ) &
            (
            python tools/make_lmdb.py data/rirs/wav.scp "${HOME}/local_lmdb/rirs/lmdb" && \
            rsync -av "${HOME}/local_lmdb/rirs/lmdb" data/rirs/lmdb
            ) &
            wait
        fi
    else
        python tools/make_lmdb.py data/musan/wav.scp data/musan/lmdb &
        python tools/make_lmdb.py data/rirs/wav.scp data/rirs/lmdb &
        wait
    fi
else
    echo "We Already have musan and rirs."
fi