#!/usr/bin/python
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
Calculate frame-level eer

"""

import os
import sys
import fire
import pandas as pd
from wedefense.metrics.detection.calculate_modules import compute_eer
from wedefense.utils.diarization.rttm_tool import get_rttm


def process_score_file(score_file,
                       gt_exist=True,
                       label2id_file='data/partialspoof/train/label2id'):
    assert os.path.isfile(label2id_file)
    # Got index of bonafide and spoof.
    label2id_dict = {}
    with open(label2id_file, "r") as f:
        for line in f:
            k, v = line.strip().split()
            label2id_dict[k] = int(v)
    bonafide_idx = label2id_dict['bonafide']
    spoof_idx = label2id_dict['spoof']

    if gt_exist:
        # Using pandas to load predicted score
        # col_names = ['reco', 'frame', 'bonafide_score', 'spoof_score'] + (['label'] if gt_exist else [])  # noqa
        col_names = ['reco', 'frame', 'bonafide_score', 'spoof_score', 'label']
        df = pd.read_csv(score_file, sep=r'\s+', header=None, names=col_names)
        # bona_cm = df[df['label'] == 'bonafide']['bonafide_score'].to_numpy()
        # spoof_cm = df[df['label'] == 'spoof']['bonafide_score'].to_numpy()
        bona_cm = df[df['label'] == str(
            bonafide_idx)]['bonafide_score'].to_numpy()
        spoof_cm = df[df['label'] == str(
            spoof_idx)]['bonafide_score'].to_numpy()
    else:
        bona_cm = []
        spoof_cm = []
        with open(score_file, 'r') as f:
            for line in f:
                frame_pred = line.strip().split()
                if len(frame_pred) < 4:
                    continus

                reco_id = frame_pred[0]
                frame_id = int(frame_pred[1])
                bonafide_score = float(frame_pred[1 + bonafide_idx])
                spoof_score = float(frame_pred[1 + spoof_idx])
                if gt_exist:
                    label = frame_pred[-1]
                elif (eval_reco2timestamps_dict):
                    raise NotImplementedError
                if (label == 'bonafide'):
                    bona_cm.append(bonafide_score)
                elif (label == 'spoof'):
                    spoof_cm.append(bonafide_score)
    return bona_cm, spoof_cm


def main(score_file,
         score_reso,
         eval_label=None,
         gt_exist=True,
         label2id_file='data/partialspoof/train/label2id',
         printout=True) -> None:
    """
    Measuring Point-based eer for frame-level predictions
    Args:
        score_file (str): Path to the frame-level prediction file.
        score_reso (int): Resolution in milliseconds, e.g., 20.
        eval_label (str): Path to RTTM file (used if gt not in score_file).
        gt_exist (bool): If True, the last column in score_file is groundtruth.
        label2id_file (str): Path to the label2id dictionary.
        printout (bool): Whether to print and save results.
    """

    print("score_file: {}".format(score_file))
    print("score_reso: {}ms".format(score_reso))
    print("eval label path: {}".format(eval_label))
    # Validity checks
    if not (gt_exist or (eval_label and os.path.isfile(eval_label))):
        raise ValueError(
            "Either score_file must include groundtruth labels or \
                          eval_label must be a valid file.")

    # TODO evaluate using input rttm
    # Load eval label if needed
    eval_reco2timestamps_dict = {}
    if (eval_label):
        # Create a map from label to the index in the logits
        if (os.path.basename(eval_label).startswith('rttm')):
            eval_reco2timestamps_dict, _ = get_rttm(eval_label)
        else:
            raise NotImplementedError(
                "Other types of label is not supported yet.")

    bona_cm, spoof_cm = process_score_file(score_file, gt_exist, label2id_file)

    # EERs of the standalone systems
    eer_cm, frr, far, thresholds, eer_threshold = compute_eer(
        bona_cm, spoof_cm)  # [0]

    if printout:
        output_file = score_file + '.point_eer'
        with open(output_file, "w") as f_res:
            f_res.write('\nCM SYSTEM\n')
            f_res.write('\tEER\t\t= {:8.9f} % '
                        '(EER for countermeasure)\n'.format(eer_cm * 100))
        os.system(f"cat {output_file}")

        print("# Point-based EER for spoof localization: \n")
        print("-eval_eer (%): {:.3f}\n".format(eer_cm * 100))
        sys.exit(0)


if __name__ == "__main__":
    fire.Fire(main)
