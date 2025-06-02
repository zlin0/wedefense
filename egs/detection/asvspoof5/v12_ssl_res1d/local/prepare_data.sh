#!/bin/bash
#
# Copyright 2025 Lin Zhang (partialspoof@gmail.com)
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
local/prepare_data.sh [PS_dir] [data]

Download partialspoof database,
and prepare data dir for partial spoof: wav.scp, utt2cls, cls2utt, utt2dur, dur2utt
"""

#
# 1. download data
# 2. wav.scp
PS_dir=$1
data=$2

# bash ./local/01_download_database.sh ${PS_dir}

for dset in train dev eval; do
  if [ ! -d ${data}/${dset} ]; then
     mkdir -p ${data}/${dset}
  fi

  find ${PS_dir}/${dset}/con_wav -name "*.wav" | awk -F"/" '{print $NF,$0}' | sort >${data}/${dset}/wav.scp
  sed -i 's/\.wav / /g' ${data}/${dset}/wav.scp
  # check row number.

  # produce utt2cls from protocols
  cut -d' ' -f2,5 ${PS_dir}/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.${dset}.trl.txt > ${data}/${dset}/utt2cls

  ./tools/utt2spk_to_spk2utt.pl ${data}/${dset}/utt2cls >${data}/${dset}/cls2utt

  #we are using wav2dur.py, but quite slow. 
  python tools/wav2dur.py ${data}/${dset}/wav.scp ${data}/${dset}/utt2dur
done

echo "Prepared data folder for partialspoof, including wav.scp, utt2cls, cls2utt"

