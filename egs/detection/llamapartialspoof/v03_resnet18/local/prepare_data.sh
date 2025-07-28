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

#
# 1. download data
# 2. wav.scp
LPS_dir=$1
data=$2

# Download data
curpath=$( pwd )
if [ ! -f ${LPS_dir}/label_R01TTS.0.a.txt ]; then
  cd ${LPS_dir}
  wget https://zenodo.org/records/14214149/files/R01TTS.0.a.tgz?download=1 -O R01TTS.0.a.tgz
  wget https://zenodo.org/records/14214149/files/label_R01TTS.0.a.txt?download=1 label_R01TTS.0.a.txt
  tar -xvf R01TTS.0.a.tgz
  echo "Finish download LlamaPartialSpoof R01TTS.0.a.tgz"
  cd $curpath
fi

mkdir -p ${data}/0a
find ${LPS_dir}/R01TTS.0.a/ -name "*.wav" | awk -F"/" '{print $NF,$0}' | sort >${data}/0a/wav.scp
sed -i 's/\.wav / /g' ${data}/0a/wav.scp
cut -d' ' -f1,3 ${LPS_dir}/label_R01TTS.0.a.txt > ${data}/0a/utt2lab
./tools/utt2lab_to_lab2utt.pl ${data}/0a/utt2lab >${data}/0a/lab2utt
python tools/wav2dur.py ${data}/0a/wav.scp ${data}/0a/utt2dur
echo "Prepared data folder for partialspoof, including wav.scp, utt2lab, lab2utt"

