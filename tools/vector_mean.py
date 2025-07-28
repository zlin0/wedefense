# Copyright (c) 2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
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

import argparse
import kaldiio
import os
import numpy as np

from tqdm import tqdm
from wespeaker.utils.utils import validate_path


def compute_vector_mean(lab2utt, xvector_scp, lab_xvector_ark):
    # read lab2utt
    lab2utt_dict = {}
    with open(lab2utt, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split(' ')
            lab2utt_dict[line[0]] = line[1:]

    utt2embs = {}
    for utt, emb in kaldiio.load_scp_sequential(xvector_scp):
        utt2embs[utt] = emb

    validate_path(lab_xvector_ark)
    lab_xvector_ark = os.path.abspath(lab_xvector_ark)
    lab_xvector_scp = lab_xvector_ark[:-3] + "scp"
    with kaldiio.WriteHelper('ark,scp:' + lab_xvector_ark + "," +
                             lab_xvector_scp) as writer:
        for lab in tqdm(lab2utt_dict.keys()):
            utts = lab2utt_dict[lab]
            mean_vec = None
            utt_num = 0
            for utt in utts:
                vec = utt2embs[utt]
                if mean_vec is None:
                    mean_vec = np.zeros_like(vec)
                mean_vec += vec
                utt_num += 1
            mean_vec = mean_vec / utt_num
            writer(lab, mean_vec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute the mean of vector')
    parser.add_argument('--lab2utt', type=str, default='', help='lab2utt file')
    parser.add_argument('--xvector_scp',
                        type=str,
                        default='',
                        help='xvector file (kaldi format)')
    parser.add_argument('--lab_xvector_ark', type=str, default='')
    args = parser.parse_args()

    compute_vector_mean(args.lab2utt, args.xvector_scp, args.lab_xvector_ark)
