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

#local/prepare_data.sh [ITW_dir] [data_dir]
#
#Download ITW database,
#and prepare data dir for partial spoof: wav.scp, utt2lab, lab2utt, utt2dur

set -xe

if [ $# -ne 2 ]; then
    echo "Usage: $0 ITW_dir data_dir"
    exit 1
fi

ITW_dir=$1 #/Path/to/spoofceleb
data_dir=$2

if [ ! -d "${ITW_dir}" ]; then
    mkdir -p "${ITW_dir}"
    bash ./01_download_database.sh "${ITW_dir}"
fi

if [ ! -d "${data_dir}" ]; then
    mkdir -p "${data_dir}"
fi

find "${ITW_dir}" -name "*.wav" | awk -v prefix="${ITW_dir}/" '{
  uttid = $0; sub(prefix, "", uttid);
  print uttid, $0
  }' |\
  sort -n > ${data_dir}/wav.scp


tail -n +2 ${ITW_dir}/meta.csv | cut -d',' -f1,3 > ${data_dir}/utt2lab
sed -i 's/bona-fide/bonafide/g' ${data_dir}/utt2lab
sed -i 's/,/ /g' ${data_dir}/utt2lab

./tools/utt2lab_to_lab2utt.pl ${data_dir}/utt2lab \
        >${data_dir}/lab2utt

#we are using wav2dur.py, but quite slow.
nj=10  # number of parallel jobs
wavscp_path=${data_dir}/wav.scp
output_dir=${data_dir}
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


echo "Prepared data folder for ITW, including wav.scp, utt2lab, lab2utt"

