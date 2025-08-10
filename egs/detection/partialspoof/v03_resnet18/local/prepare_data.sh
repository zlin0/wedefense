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
This script downloads the PartialSpoof database and prepares the data directory
  in a Kaldi-style format.
  It generates: wav.scp, utt2lab, lab2utt, and utt2dur.

Usage:
  local/prepare_data.sh <path/to/partialspoof/database> <output/data/dir>
"""

set -e -u -o pipefail # Fail on error, undefined variable, or pipe failure

PS_dir=$1
data=$2

echo "Stage 0: Downloading PartialSpoof database if not present..."
local/01_download_database.sh "${PS_dir}"

for dset in train dev eval; do
(
  out_dir="${data}/${dset}"
  echo "Processing ${dset} set, outputting to ${out_dir}"
  mkdir -p "${out_dir}"

  # Create wav.scp: <utt_id> <path_to_wav>
  find "${PS_dir}/${dset}/con_wav" -name "*.wav" | awk -F"/" '{print $NF,$0}' | sort > "${out_dir}/wav.scp"
  # The sed command removes the .wav extension from the utterance ID.
  sed -i 's/\.wav / /' "${out_dir}/wav.scp"

  # Create utt2lab from protocols: <utt_id> <label>
  cut -d' ' -f2,5 "${PS_dir}/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.${dset}.trl.txt" > "${out_dir}/utt2lab"

  # Create lab2utt: <label> <utt_id_1> <utt_id_2> ...
  ./tools/utt2lab_to_lab2utt.pl "${out_dir}/utt2lab" > "${out_dir}/lab2utt"

  # Create utt2dur: <utt_id> <duration_in_seconds>
  # Use the parallel script which prefers soxi for speed.
  # We are running 3 jobs in parallel (train, dev, eval), so we divide the total cores by 3 (train/dev/eval).
  nj=$(( $(nproc) / 3 )) # we have three sets.
  [ $nj -lt 1 ] && nj=1 # Ensure at least 1 job is used, even on machines with < 3 cores.
  tools/wav_to_duration.sh --nj "${nj}" "${out_dir}/wav.scp" "${out_dir}/utt2dur"
) &
done
wait

echo "Successfully prepared data in ${data}"
