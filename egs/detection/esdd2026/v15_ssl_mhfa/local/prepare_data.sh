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

#local/prepare_data.sh [EnvSDD_dir] [data_dir]
#
#Download EnvSDD database,
#and prepare data dir for partial spoof: wav.scp, utt2lab, lab2utt, utt2dur

set -xe

EnvSDD_dir=$1 #/Path/to/EnvSDD
data_dir=$2

DSETs=(development dev_track2 eval_track1 eval_track2)

if [ ! -d ${EnvSDD_dir} ]; then
    mkdir -p ${EnvSDD_dir}
    bash ./01_download_database.sh ${EnvSDD_dir} 
fi

for i in "${!DSETs[@]}"; do
  dset=${DSETs[$i]}	

  if [ ! -d ${data_dir}/${dset} ]; then
     mkdir -p ${data_dir}/${dset}
  fi

  echo ${EnvSDD_dir}/$dset

  find "${EnvSDD_dir}/${dset}/" -name "*.wav" | \
  awk -v prefix="${EnvSDD_dir}/${dset}/" '{
      rel = $0; sub(prefix, "", rel);
      print rel, $0;
  }' | sort > "${data_dir}/${dset}/wav.scp"
#
  if [[ "${dset}" == "development" ]]; then
    grep 'real_audio' ${data_dir}/${dset}/wav.scp | awk '{print $1, "real"}' > ${data_dir}/$dset/utt2lab 
    grep 'fake_audio' ${data_dir}/${dset}/wav.scp | awk '{print $1, "fake"}' >> ${data_dir}/$dset/utt2lab 
  elif [[ "${dset}" == "dev_track2" ]]; then
    csv_file="${EnvSDD_dir}/dev_track2.csv"
    awk -F',' 'NR>1 && $1!="" {print $1 " " $2}' "${csv_file}" >  ${data_dir}/$dset/short_utt2lab 

    awk '
      NR==FNR { LAB[$1]=$2}
      NR!=FNR {
        n = split($1, a, "/");    
        fname = a[n];
        if (fname in LAB) {
          label = LAB[fname];
          print $1, label;
        }
      }
    ' "${data_dir}/${dset}/short_utt2lab" "${data_dir}/${dset}/wav.scp" \
      > "${data_dir}/${dset}/utt2lab"
  else
     echo "No label avalible for "${dset}
  fi

  if [ ! -e ${data_dir}/${dset}/utt2lab ]; then
    ./tools/utt2lab_to_lab2utt.pl ${data_dir}/${dset}/utt2lab \
  	  >${data_dir}/${dset}/lab2utt
  fi
#
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
#



# split train/val for development:
# Expected:
# ${EnvSDD_dir}/dev_track1.csv            # track1 的CSV
# ${EnvSDD_dir}/dev_track2.csv            # track2 的CSV
# ${data_dir}/development/{wav.scp,utt2lab,utt2dur}  # track1 源数据
# ${data_dir}/dev_track2/{wav.scp,utt2lab,utt2dur}   # track2 源数据
#
for idx in 1 ; do

  if [[ "${idx}" -eq 1 ]]; then
    csv_file="${EnvSDD_dir}/dev_track1.csv"   
    pa_dir="development"
    output_dir="${data_dir}/${pa_dir}"
    awk -F',' 'NR>1 && $1!="" {print $2"/"$1 " " $NF}' "${csv_file}" > "${output_dir}/real_wav2dset"
    awk '
      NR==FNR { SET[$1]=$2}
      NR!=FNR {
        n = split($1, a, "/");    
        fname = a[n-1]"/"a[n];
        if (fname in SET) {
          label = SET[fname];
          print $1, label;
        }
      }
    ' "${output_dir}/real_wav2dset" "${data_dir}/development/wav.scp" \
      > "${data_dir}/development/wav2dset"
  else
    csv_file="${EnvSDD_dir}/dev_track2.csv"
    pa_dir="dev_track2"
    output_dir="${data_dir}/${pa_dir}"
    awk -F',' 'NR>1 && $1!="" {print $1 " " $NF}' "${csv_file}" > "${output_dir}/wav2dset"
  fi

  # split {wav.scp, utt2lab, utt2dur} into（train/val）according to metadate
  # according to ${output_dir}/wav2dset
  if [ ! -d ${data_dir}/train_track${idx} ]; then
    rm -r ${data_dir}/train_track${idx}
    mkdir ${data_dir}/train_track${idx}
  fi
  if [ ! -d ${data_dir}/train_track${idx} ]; then
    rm -r ${data_dir}/validation_track${idx}
    mkdir ${data_dir}/validation_track${idx}
  fi
  for data_file in wav.scp utt2lab utt2dur; do
    src_file="${data_dir}/${pa_dir}/${data_file}"

    ##if [[ "${pa_dir}" == "development" ]]; then
    #if [[ "${idx}" -eq 1 ]]; then
    #  awk -v out="${data_dir}" -v idx=${idx} -v df="${data_file}" '
    #    (NR==FNR) { SET[$1]=$2 }
    #    (NR!=FNR) {
    #      line=$0; n=split(line, a, "/")
    #      utt=a[n-1]"/"a[n]
    #      subset=SET[utt]
    #      if (subset!="") {
    #        dir=out "/" subset "_track" idx
    #        path=dir "/" df
    #        print $0 >> path
    #      }
    #    }
    #  ' "${output_dir}/wav2dset" "${src_file}"
    #else
      awk -v out="${data_dir}" -v idx=${idx} -v df="${data_file}" '
        (NR==FNR) { SET[$1]=$2 }
        (NR!=FNR) {
          utt=$1
          subset=SET[utt]
          if (subset!="") {
            dir=out "/" subset "_track" idx
            path=dir "/" df
            print $0 >> path
          }
        }
      ' "${output_dir}/wav2dset" "${src_file}"
     # fi

  done

  for sub in train validation; do
    sub_dir="${data_dir}/${sub}_track${idx}"
    ./tools/utt2lab_to_lab2utt.pl "${sub_dir}/utt2lab" > "${sub_dir}/lab2utt"
  done
done


echo "Prepared data folder for EnvSDD, including wav.scp, utt2lab, lab2utt"
