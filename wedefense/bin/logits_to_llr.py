#!/usr/bin/env python

# Copyright (c) 2024 Johan Rohdin (rohdin@fit.vutbr.cz)
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

import fire
import kaldiio
import numpy as np
from wedefense.utils.file_utils import read_table
from wedefense.utils.utils import spk2id
from scipy.special import logsumexp
import os.path

# import torch
# from wedefense.utils.utils import parse_config_or_kwargs
# from wedefense.models.projections import get_projection
# import os.path
# from scipy.special import softmax


def main(logits_scp_path, training_counts, train_label, pi_spoof):

    print("logits_scp_path {}".format(logits_scp_path))

    utt = []
    logits = []
    for k, v in kaldiio.load_scp_sequential(logits_scp_path):
        utt.append(k)
        logits.append(v)
    logits = np.vstack(logits)
    print(logits.shape)

    # Create a mapping from label to prior (relative frequency) in training data
    training_counts_info = np.genfromtxt(training_counts,
                                         dtype=str,
                                         delimiter=" ")
    training_counts = training_counts_info[:, 1].astype(np.float64)
    training_counts_labs = training_counts_info[:, 0]
    print(training_counts_labs)

    training_priors = training_counts / np.sum(training_counts)
    lab2training_prior = dict(zip(training_counts_labs, training_priors))

    # Create a map from label to the index in the logits
    train_utt_spk_list = read_table(train_label)
    lab2id_dict = spk2id(train_utt_spk_list)

    bonafide_idx = []
    bonafide_priors = []
    spoof_idx = []
    spoof_priors = []

    for k, v in lab2id_dict.items():
        print(k)
        if "bonafide" in k:
            bonafide_idx.append(v)
            bonafide_priors.append(lab2training_prior[k])
        else:
            spoof_idx.append(v)
            spoof_priors.append(lab2training_prior[k])

    ll_bonafide = logsumexp(logits[:, bonafide_idx] -
                            np.log(np.array(bonafide_priors)),
                            axis=1)
    ll_spoof = logsumexp(logits[:, spoof_idx] - np.log(np.array(spoof_priors)),
                         axis=1)
    # there should be + log(1/I), but I=1, so +0.

    print(np.array(bonafide_priors))
    print(np.array(spoof_priors))
    print(np.log(np.array(bonafide_priors)))
    print(np.log(np.array(spoof_priors)))

    # ll_bonafide = logsumexp(logits[:,bonafide_idx] , axis=1)
    # ll_spoof    = logsumexp(logits[:,spoof_idx] , axis=1)

    llr = ll_bonafide - ll_spoof
    nan_idx = np.where(np.isnan(llr))
    if len(nan_idx[0]):
        # print( "Warning: {} LLRs where nan! Set to mean of remaining LLRs".format(len(nan_idx[0])) )   # noqa
        # llr[nan_idx]= np.mean(llr[np.where(~np.isnan(llr))])

        # The above would probably violate the competition rules.
        print("Warning: {} LLRs where nan! Set to 0".format(len(nan_idx[0])))
        llr[nan_idx] = 0

    out_path = os.path.dirname(logits_scp_path)

    # with kaldiio.WriteHelper('ark,scp:' + out_path + "/llr.ark," + out_path + "/llr.scp") as writer:  # noqa
    #    for i, utt in enumerate(utt):
    #        writer(utt, np.array(llr[i].astype(np.float64)))

    with open(out_path + "/llr.txt", "w") as f:
        f.write("filename\tcm-score\n")
        for i, u in enumerate(utt):
            f.write("{}\t{}\n".format(u, llr[i]))


if __name__ == "__main__":
    fire.Fire(main)
