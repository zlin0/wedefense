#!/usr/bin/env python

# Copyright (c) 2025 Johan Rohdin (rohdin@fit.vutbr.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
apply_calibration_fusion_model.py
Todo: The code for reading scores etc. is the same as in
      apply_calibration_fusion_model.py. Consider to put it in a common place.
"""

import glob
import sys
import fire
import pandas as pd
import torch as T
import numpy as np

from wedefense.postprocessing.models import LogisticRegression

pd.set_option('future.no_silent_downcasting', True)


def main(model_load_path, score_dir, new_scores_file, method='LR'):

    # Search for scores
    print("Using scores in {} for calibration / fusion.".format(score_dir))

    systems = glob.glob("*", root_dir=score_dir)
    n_sys = len(systems)

    if n_sys <= 0:
        print("ERROR: No scores found!")
        sys.exit(2)
    else:
        print("Found {} systems: {}".format(n_sys, systems))

    # Read in scores and make a dataframe.
    for i in range(len(systems)):
        df_tmp = pd.read_csv(score_dir+'/'+systems[i]+'/llr.txt', sep='\t')
        df_tmp = df_tmp.rename(columns={'cm-score':
                                        'cm-score'
                                        + str(i)}).set_index('filename')
        # To consider: Would it better to put the system's name in the column
        #              names? (For clearer debugging / analysis.)
        if i == 0:
            df = df_tmp
        else:
            df = df.join(df_tmp, on='filename')

    filenames = df.index
    df = df.to_numpy().astype(np.float32)

    # Initialize and load the model. TODO: Code should support more models.
    if method == 'LR':
        device = T.device("cpu")
        model = LogisticRegression(n_sys=n_sys, p_tar=0.5).to(device)
        model.load_state_dict(T.load(model_load_path))
        model.eval()

        # Creat the scores. Setting p_tar=0.5 means we will get LLRs. It
        # overrides the p_tar of the model.
        llr = model(T.tensor(df), p_tar=0.5).detach().numpy()

    elif method == 'min_max_avg':
        aff_trans = np.load(model_load_path, allow_pickle=False)
        llr = df.dot(aff_trans[0:-1]) + aff_trans[-1]

    with open(new_scores_file, "w") as g:
        g.write("filename\tcm-score\n")
        for i, n in enumerate(filenames):
            g.write("{}\t{}\n".format(n, llr[i].squeeze()))


if __name__ == "__main__":
    fire.Fire(main)
