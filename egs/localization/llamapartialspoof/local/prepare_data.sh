#!/bin/bash
#
# Copyright 2025 Hieu-Thi Luong (contact@hieuthi.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
local/prepare_data.sh [LPS_dir] [data]

Download LlamaPartialSpoof database,
and prepare data dir for partial spoof evaluation: wav.scp, utt2lab, lab2utt, utt2dur, dur2utt
"""

set -ex

#
# 1. download data
# 2. wav.scp
LPS_dir=$1
data=$2

FILEs="label_R01TTS.0.a.txt
label_R01TTS.0.b.txt
LICENSE.txt
metadata_crossfade.csv
R01TTS.0.a.tgz
R01TTS.0.b.tgz
README.txt
"

# - R01TTS.0.a contains bonafide, fully fake (TTS001--006), partially fake (TTS001--006) using crossfade
# - R01TTS.0.b contains partially fake (TTS001--006) using cut/paste or overlap/add

# Download data
curpath=$( pwd )
# if [ ! -f ${LPS_dir}/label_R01TTS.0.a.txt ]; then
#   cd ${LPS_dir}
#   for file in ${FILEs}; do
#     wget https://zenodo.org/records/14214149/files/${file}?download=1 -O ${file}
#     echo "Finish download LlamaPartialSpoof R01TTS.0.a.tgz"
#   done
#   tar -zxvf R01TTS.0.a.tgz
#   tar -zxvf R01TTS.0.b.tgz
#   cd $curpath
# fi

echo "Start to prepare data for llamapartialspoof"

# mkdir -p ${data}/0a
# find ${LPS_dir}/R01TTS.0.a/ -name "*.wav" | awk -F"/" '{print $NF,$0}' | sort > ${data}/0a/wav.scp
# sed -i 's/\.wav / /g' ${data}/0a/wav.scp
# cut -d' ' -f1,3 ${LPS_dir}/label_R01TTS.0.a.txt > ${data}/0a/utt2lab
# ./tools/utt2lab_to_lab2utt.pl ${data}/0a/utt2lab > ${data}/0a/lab2utt
# # python tools/wav2dur.py ${data}/0a/wav.scp ${data}/0a/utt2dur

dset="0a"
#we are using wav2dur.py, but quite slow.
  nj=10  # number of parallel jobs
  wavscp_path=${data}/${dset}/wav.scp
  output_dir=${data}/${dset}
  split_dir=${output_dir}/split_wavscp

  mkdir -p $split_dir
  split -n l/$nj -d --additional-suffix=.scp "$wavscp_path" "$split_dir/wav_part_"

  # run wav2dur.py in parallel
  for i in $(seq -f "%02g" 0 $((nj - 1))); do
    python tools/wav2dur.py "$split_dir/wav_part_${i}.scp" "$split_dir/utt2dur_${i}" &
  done

  wait  # wait for all jobs to finish

  # merge all utt2dur files
  cat "$split_dir"/utt2dur_* > "$output_dir/utt2dur"

  # optional: clean up
  rm -r "$split_dir"

  echo "Done: utt2dur saved to $output_dir/utt2dur"


echo "Prepared data folder for llamapartialspoof, including wav.scp, utt2lab, lab2utt"

