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

#local/prepare_data.sh [ASVspoof5_dir] [data_dir]
#
#Download ASVspoof5 database,
#and prepare data dir for partial spoof: wav.scp, utt2cls, cls2utt, utt2dur

set -xe

ASVspoof5_dir=$1
data_dir=$2

DSETs=(T D E_eval)
DSETs_full=(train dev eval)

if [ ! -d ${ASVspoof5_dir} ]; then
    mkdir -p ${ASVspoof5_dir}
    bash ./01_download_database.sh ${ASVspoof5_dir}
fi

for i in "${!DSETs[@]}"; do
  dset=${DSETs[$i]}
  dset_full=${DSETs_full[$i]}

  if [ ! -d ${data_dir}/flac_${dset}_all ]; then
     mkdir -p ${data_dir}/flac_${dset}_all
  fi

  find ${ASVspoof5_dir}/flac_${dset}/ -name "*.flac" | awk -F"/" '{print $NF,$0}' |\
          sort > ${data_dir}/flac_${dset}_all/wav.scp
  sed -i 's/\.flac / /g' ${data_dir}/flac_${dset}_all/wav.scp
  # check row number.


  # produce utt2cls from protocols
  if [ "$dset" = "T"  ]; then
    cut -d' ' -f2,9 ${ASVspoof5_dir}/ASVspoof5.${dset_full}.tsv \
	    > ${data_dir}/flac_${dset}_all/utt2cls
  else
    cut -d' ' -f2,9 ${ASVspoof5_dir}/ASVspoof5.${dset_full}.track_1.tsv \
	    > ${data_dir}/flac_${dset}_all/utt2cls
  fi

  ./tools/utt2spk_to_spk2utt.pl ${data_dir}/flac_${dset}_all/utt2cls \
	  >${data_dir}/flac_${dset}_all/cls2utt

  #we are using wav2dur.py, but quite slow.
  python tools/wav2dur.py ${data_dir}/flac_${dset}_all/wav.scp ${data_dir}/flac_${dset}_all/utt2dur


  # Extract list for track1 - deepfake detection
  if [ "$dset" = "T"  ]; then
      if [ ! -e ${data_dir}/flac_${dset} ]; then
          ln -s flac_${dset}_all ${data_dir}/flac_${dset}
      fi
  else
      if [ ! -d ${data_dir}/flac_${dset} ]; then
            mkdir -p ${data_dir}/flac_${dset}
      fi
     for fname in wav.scp utt2cls cls2utt utt2dur; do
         if [ ! -f ${data_dir}/flac_${dset}/${fname} ]; then
		 awk '(NR==FNR){FILE[$2]}(NR!=FNR){
			 if($1 in FILE){print}
		 }' ${ASVspoof5_dir}/ASVspoof5.${dset_full}.track_1.tsv \
			 ${data_dir}/flac_${dset}_all/${fname} \
			 > ${data_dir}/flac_${dset}/${fname}
	 fi
     done
     ./tools/utt2spk_to_spk2utt.pl ${data_dir}/flac_${dset}/utt2cls \
	  >${data_dir}/flac_${dset}/cls2utt
  fi
done

echo "Prepared data folder for partialspoof, including wav.scp, utt2cls, cls2utt"

