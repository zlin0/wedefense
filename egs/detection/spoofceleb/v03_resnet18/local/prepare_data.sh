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

#local/prepare_data.sh [SpoofCeleb_dir] [data_dir]
#
#Download SpoofCeleb database,
#and prepare data dir for partial spoof: wav.scp, utt2cls, cls2utt, utt2dur

set -xe

SpoofCeleb_dir=$1 #/Path/to/spoofceleb
data_dir=$2

DSETs=(train development evaluation)

if [ ! -d ${SpoofCeleb_dir} ]; then
    mkdir -p ${SpoofCeleb_dir}
    bash ./01_download_database.sh ${SpoofCeleb_dir} 
fi

for i in "${!DSETs[@]}"; do
  dset=${DSETs[$i]}	


  if [ ! -d ${data_dir}/${dset} ]; then
     mkdir -p ${data_dir}/${dset}
  fi

  find ${SpoofCeleb_dir}/flac/${dset}/ -name "*.flac" | awk -v prefix="${SpoofCeleb_dir}/flac/$dset" '{
    rel = $0; sub(prefix, "", rel);
    print rel, $0
    }' |\
    sort > ${data_dir}/${dset}/wav.scp
  # To make sure we downloaded enough flac, we didn't use the command below.
  awk -v prefix=${SpoofCeleb_dir}/flac/$dset -F',' '(NR>1){print $1, prefix"/"$1}' \
  	  ${SpoofCeleb_dir}/metadata/$dset.csv  > ${data_dir}/$dset/wav.scp


  # check row number.
  wav_count=$(wc -l < "${data_dir}/${dset}/wav.scp")
  csv_count=$(wc -l < "${SpoofCeleb_dir}/metadata/$dset.csv")
  csv_count=$((csv_count - 1))
  if [ "$wav_count" -eq "$csv_count" ]; then
    echo "wav.scp has correct wavforms."
  else
    echo "${dset}/wav.scp does NOT have correct row. Please have a check. It has $(wc -l < ${data_dir}/${dset}/wav.scp) lines."
  fi

  # produce utt2cls from protocols
  awk -F',' '(NR>1){if($3=="a00"){label="bonafide"}else{label="spoof"}
      print($1, label)
  }' ${SpoofCeleb_dir}/metadata/$dset.csv > ${data_dir}/$dset/utt2cls

  ./tools/utt2spk_to_spk2utt.pl ${data_dir}/${dset}/utt2cls \
	  >${data_dir}/${dset}/cls2utt

  #we are using wav2dur.py, but quite slow. 
  nj=10  # number of parallel jobs
  wavscp_path=${data_dir}/${dset}/wav.scp
  output_dir=${data_dir}/${dset}
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



done

echo "Prepared data folder for SpoofCeleb, including wav.scp, utt2cls, cls2utt"

