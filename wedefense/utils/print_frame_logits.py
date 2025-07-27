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
Print logits for better analyses.

<reco_id> <frame_id> <bonafide_score> <spoof_score>

"""

import fire
import kaldiio
import json
import numpy as np
from tqdm import tqdm
from wedefense.utils.diarization.rttm_tool import get_rttm
import os.path


def convert_score_reso(logits: np.ndarray, src_reso: int,
                       tgt_reso: int) -> np.ndarray:
    """
    Convert frame-level logits from one temporal resolution to another.

    Args:
        logits (np.ndarray): shape (T, C),
            where T is number of frames and C is number of classes
        src_reso (int): source frame resolution in ms (e.g. 20)
        tgt_reso (int): target frame resolution in ms (e.g. 40)

    Returns:
        np.ndarray: converted logits with new resolution
    """
    assert logits.ndim == 2, "logits must be 2D array (T, C)"
    T, C = logits.shape

    if src_reso == tgt_reso:
        return logits

    ratio = tgt_reso / src_reso

    if ratio > 1:
        # Downsample: average-pool to fewer frames
        new_T = int(np.ceil(T / ratio))
        new_logits = []
        for i in range(new_T):
            start = int(i * ratio)
            end = int(min((i + 1) * ratio, T))
            pooled = logits[start:end].mean(axis=0)
            new_logits.append(pooled)
        return np.stack(new_logits)

    else:
        # Upsample: repeat frames to get higher resolution
        repeat_factor = int(src_reso / tgt_reso)
        return np.repeat(logits, repeat_factor, axis=0)


def main(logits_scp_path, score_reso, eval_reso, train_label, eval_label=None):
    """
    Args:
        logits_scp_path (str): path to logits.scp
        score_reso (int): resolution of logits in ms, e.g., 20
        eval_reso (int): resolution to convert to, e.g., 40
        train_label (str): path to RTTM-style label
        eval_label (str): optional, RTTM for evaluation
    """

    print("logits_scp_path {}".format(logits_scp_path))
    print("score_reso {}".format(score_reso))
    print("eval_reso {}".format(eval_reso))
    print("train_label {}".format(train_label))

    # Create a map from label to the index in the logits
    if (os.path.basename(train_label).startswith('rttm')):
        _, label2id_dict = get_rttm(train_label)
    else:
        raise NotImplementedError("Other types of label is not supported yet.")
    # Got index of bonafide and spoof.
    bonafide_idx, spoof_idx = label2id_dict['bonafide'], label2id_dict['spoof']

    if (eval_label):
        if (os.path.basename(eval_label).startswith('rttm')):
            eval_reco2timestamps_dict, _ = get_rttm(eval_label)

    utts_lst = []
    logits_lst = []

    out_path = os.path.dirname(logits_scp_path)
    with open(os.path.join(out_path, f"logits_frame_{eval_reso}ms.txt"),
              "w") as f:
        for utt, logit in tqdm(kaldiio.load_scp_sequential(logits_scp_path)):
            new_logit = convert_score_reso(logit, score_reso, eval_reso)
            for fdx, logit_frame in enumerate(new_logit):
                line = "{}\t{}\t{}\t{}".format(utt, fdx,
                                               logit_frame[bonafide_idx],
                                               logit_frame[spoof_idx])

                if (eval_label):
                    t_sec = fdx * (eval_reso / 1000.0
                                   )  # second of the current frame
                    label_segments = eval_reco2timestamps_dict.get(utt, [])
                    label_frame = "unknown"
                    for seg_label, st, et in label_segments:
                        if st <= t_sec < et:
                            label_frame = label2id_dict[seg_label]
                            # label_frame = seg_label  # save bonafide/spoof
                            break
                    line += f"\t{label_frame}"
                f.write(line + "\n")

            utts_lst.append(utt)
            logits_lst.append(new_logit)

    with open(os.path.join(out_path, "label2id.json"), "w") as f:
        json.dump(label2id_dict, f, indent=2)

    all_logits = np.vstack(logits_lst)
    print("Final logits shape:", all_logits.shape)
    print(f"Written frame-level scores to {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
