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


# This script prepares the data directory for the ASVspoof 2024 (ASVspoof5)
# dataset in a Kaldi-style format.
# It generates: wav.scp, utt2lab, lab2utt, and utt2dur for train, dev, and eval sets.

# Usage:
#   local/prepare_data.sh <path/to/asvspoof5> <output/data/dir>


set -ex -o pipefail # Fail on error, undefined variable, or pipe failure

ASVspoof5_dir=$1
data_dir=$2

DSETs=(T D E_eval)
DSETs_full=(train dev eval)

if [ ! -d "${ASVspoof5_dir}" ]; then
    echo "ASVspoof5 directory not found at ${ASVspoof5_dir}. Attempting to download."
    mkdir -p "${ASVspoof5_dir}"
    bash ./local/01_download_database.sh "${ASVspoof5_dir}"
fi

for i in "${!DSETs[@]}"; do
  dset_short=${DSETs[$i]}
  dset_full=${DSETs_full[$i]}
  out_dir_all="${data_dir}/flac_${dset_short}_all"
  out_dir_final="${data_dir}/flac_${dset_short}"

  echo "Processing ${dset_full} set (from ${dset_short}), outputting to ${out_dir_final}"
  mkdir -p "${out_dir_all}"

  # Create wav.scp for all files in the original directory
  find "${ASVspoof5_dir}/flac_${dset_short}/" -name "*.flac" | awk -F"/" '{print $NF,$0}' | \
          sort > "${out_dir_all}/wav.scp"
  sed -i 's/\.flac / /' "${out_dir_all}/wav.scp"

  # produce utt2lab from protocols <uttid> <label>
  if [ "${dset_short}" = "T"  ]; then
    # Train set has a single protocol file
    cut -d' ' -f2,9 "${ASVspoof5_dir}/ASVspoof5.${dset_full}.tsv" \
	    > "${out_dir_all}/utt2lab"
  else
    # Dev and Eval sets have track-specific protocols
    cut -d' ' -f2,9 "${ASVspoof5_dir}/ASVspoof5.${dset_full}.track_1.tsv" \
	    > "${out_dir_all}/utt2lab"
  fi

  ./tools/utt2lab_to_lab2utt.pl "${out_dir_all}/utt2lab" \
	  > "${out_dir_all}/lab2utt"

  # Create utt2dur: <utt_id> <duration_in_seconds>
  # The outer loop runs sequentially, so we can use all available cores for each set.
  nj=$(nproc)
  tools/wav_to_duration.sh --nj "${nj}" "${out_dir_all}/wav.scp" "${out_dir_all}/utt2dur"

  # For ASVspoof5, we are interested in Track 1 (deepfake detection).
  # The train set uses all data, while dev and eval sets are filtered.
  if [ "${dset_short}" = "T"  ]; then
      # For the training set, just create a symlink
      ln -snf "${dset_full}_all" "${out_dir_final}"
  else
      # For dev and eval, filter the files based on the track 1 protocol file.
      mkdir -p "${out_dir_final}"
      track1_keys="${ASVspoof5_dir}/ASVspoof5.${dset_full}.track_1.tsv"
      for fname in wav.scp utt2lab lab2utt utt2dur; do
          awk '
            # Load the keys from the protocol file into an array
            NR==FNR { keys[$2]; next }
            # For the other files, print the line if the first field (utt_id) is in our keys
            ($1 in keys) { print }
          ' "${track1_keys}" "${out_dir_all}/${fname}" > "${out_dir_final}/${fname}"
      done
      # Re-generate lab2utt for the filtered set
      ./tools/utt2lab_to_lab2utt.pl "${out_dir_final}/utt2lab" > "${out_dir_final}/lab2utt"
  fi
  ln -s flac_${dset_short} ${data_dir}/${dset_full}
done

echo "Successfully prepared data for ASVspoof5 in ${data
