#``!/usr/bin/env python
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


#TODO merge with make_rttm.py
import argparse
from collections import defaultdict

def process_rttm(rttm_file, label_map, output_file):
    segments = defaultdict(list)

    # Step 1: Read and map labels
    with open(rttm_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue  # Skip lines with not enough value.
            utt = parts[1]
            start = float(parts[3])
            dur = float(parts[4])
            if(dur < 1e-3): # Skip extremely short or zero-duration segments
                continue
            label = parts[7]

            mapped_label = label_map[label]
            segments[utt].append((start, dur, mapped_label))

    # Step 2: Merge adjacent segments with the same label
    with open(output_file, 'w') as f_out:
        for utt, segs in segments.items():
            segs.sort()
            merged = []
            for seg in segs:
                if not merged:
                    merged.append(list(seg))
                else:
                    last = merged[-1]
                    if abs((last[0] + last[1]) - seg[0]) < 1e-6 and last[2] == seg[2]:
                        last[1] += seg[1]
                    else:
                        merged.append(list(seg))

            for s, d, l in merged:
                if(d < 1e-3): # Skip extremely short or zero-duration segments
                    continue
                f_out.write(f"SPEAKER {utt} 1 {s:7.3f} {d:7.3f} <NA> <NA> {l} <NA> <NA>\n")

def main():
    parser = argparse.ArgumentParser(description="Map and clean RTTM labels")
    parser.add_argument("--input_rttm", required=True, help="Path to input RTTM file")
    parser.add_argument("--map_file", required=True, help="Path to label mapping file")
    parser.add_argument("--output_rttm", required=True, help="Path to save cleaned RTTM")
    args = parser.parse_args()

    label_map = dict([line.split() for line in open(args.map_file)])
    process_rttm(args.input_rttm, label_map, args.output_rttm)

if __name__ == "__main__":
    main()



