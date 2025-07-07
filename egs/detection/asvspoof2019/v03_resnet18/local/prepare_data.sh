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
local/prepare_data.sh [ASVspoof2019_dir] [data]

Download ASVspoof2019 database,
and prepare data dir for partial spoof: wav.scp, utt2cls, cls2utt, utt2dur, dur2utt
Note that we only implemented LA
"""
set -ex

#
# 1. download data
# 2. wav.scp
ASVspoof2019_dir=$1
data=$2

bash ./local/01_download_database.sh ${ASVspoof2019_dir}

for dset in train dev eval; do
  if [ ! -d ${data}/${dset} ]; then
     mkdir -p ${data}/${dset}
  fi

  find ${ASVspoof2019_dir}/LA/ASVspoof2019_LA_${dset}/flac/ -name "*.flac" |\
	  awk -F"/" '{print $NF,$0}' | sort >${data}/${dset}/wav.scp
  sed -i 's/\.flac / /g' ${data}/${dset}/wav.scp
  # check row number.

  # produce utt2cls from protocols
  if [ "${dset}" == "train"  ]; then
      cut -d' ' -f2,5 ${ASVspoof2019_dir}/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.${dset}.trn.txt > ${data}/${dset}/utt2cls
  else
      cut -d' ' -f2,5 ${ASVspoof2019_dir}/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.${dset}.trl.txt > ${data}/${dset}/utt2cls
  fi

  ./tools/utt2spk_to_spk2utt.pl ${data}/${dset}/utt2cls >${data}/${dset}/cls2utt

  #we are using wav2dur.py, but quite slow. 
  python tools/wav2dur.py ${data}/${dset}/wav.scp ${data}/${dset}/utt2dur
done

echo "Prepared data folder for ASVspoof2019, including wav.scp, utt2cls, cls2utt"

