#!/usr/bin/env python

# Copyright (c) 2025 Lin Zhang (partialspoof@gmail.com)
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
Convert frame-level prediction to rttm
frame-level prediction has format as:

<reco_id> <frame_id> <bonafide_score> <spoof_score> [<ground_truth>]

"""

import argparse
import kaldiio
from tqdm import tqdm
import json
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert frame-level logits to RTTM "
        "by argmax post-processing.")

    parser.add_argument('--logits_scp_path',
                        type=str,
                        required=True,
                        help="Path to the Kaldi-style logits.scp file.")
    parser.add_argument('--score_reso',
                        type=int,
                        required=True,
                        help="Resolution of logits in milliseconds "
                        "(e.g., 20 means each frame is 20ms).")
    parser.add_argument('--output_rttm',
                        type=str,
                        required=True,
                        help="Output RTTM file path.")
    parser.add_argument('--label2id_file',
                        type=str,
                        default='data/partialspoof/train/label2id',
                        help="Path to label2id file mapping labels to indices "
                        "(e.g., {'bonafide': 0, 'spoof': 1}).")
    # parser.add_argument('--label_exist',
    #                     type=bool,
    #                     default=True,
    #     help="The input logit file contain labels (at the last column)?")

    return parser.parse_args()


def get_label2id(label2id_file):
    assert os.path.isfile(label2id_file)
    if (label2id_file.lower().endswith('.json')):
        with open(label2id_file, 'r', encoding='utf-8') as f:
            label2id_dict = json.load(f)  # Load JSON data
    else:
        label2id_dict = {}
        with open(label2id_file, "r") as f:
            for line in f:
                k, v = line.strip().split()
                label2id_dict[k] = int(v)
    return label2id_dict


def logits_to_rttm(logits_scp_path, score_reso, output_rttm, label2id_file):
    print("logits_scp_path:", logits_scp_path)
    print("score_reso:", score_reso)
    print("output_rttm:", output_rttm)
    print("label2id_file:", label2id_file)

    # Got index of bonafide and spoof.
    label2id_dict = get_label2id(label2id_file)

    bonafide_idx = label2id_dict['bonafide']
    spoof_idx = label2id_dict['spoof']

    frame_shift = score_reso / 1000.0  # Convert ms to seconds

    with open(output_rttm, "w") as f_out:
        for utt, scores in tqdm(kaldiio.load_scp_sequential(logits_scp_path),
                                desc="Converting"):
            preds = scores.argmax(axis=1)  # Frame-level predictions (0 or 1)

            if len(preds) == 0:
                continue

            start_idx = 0
            current_label = preds[0]

            # Iterate through predictions to find segment boundaries
            for idx in range(1, len(preds)):
                if preds[idx] != current_label:
                    start_time = start_idx * frame_shift
                    duration = (idx - start_idx) * frame_shift
                    label = 'bonafide' if current_label == bonafide_idx else 'spoof'
                    f_out.write(
                        f"SPEAKER {utt} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {label} <NA> <NA>\n"
                    )

                    # Update start for new segment
                    start_idx = idx
                    current_label = preds[idx]

            # Write final segment
            start_time = start_idx * frame_shift
            duration = (len(preds) - start_idx) * frame_shift
            label = 'bonafide' if current_label == bonafide_idx else 'spoof'
            f_out.write(
                f"SPEAKER {utt} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {label} <NA> <NA>\n"
            )


if __name__ == "__main__":
    args = parse_arguments()
    logits_to_rttm(args.logits_scp_path, args.score_reso, args.output_rttm,
                   args.label2id_file)
