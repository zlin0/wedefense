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
train_calibration_fusion_model.py
Trains calibration and/or fusion of scores for binary decisions.
Partially Based on
* https://visualstudiomagazine.com/articles/2021/06/23/logistic-regression
  -pytorch.aspx
* https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/bin/
  /score_calibration.py
"""

import glob
import sys
import fire
import pandas as pd
import torch as T
import numpy as np

from wedefense.postprocessing.models import LogisticRegression

pd.set_option('future.no_silent_downcasting', True)


# This function should be used for many models
def train_with_BFGS(model, train_set, p_tar=-1, max_iter=20, lr=0.01):
    model.train()
    loss_func = T.nn.BCEWithLogitsLoss(reduction='none')  # We have logits

    def weighted_loss_func(P, Y, W):
        return T.mean(W*loss_func(P, Y))

    opt = T.optim.LBFGS(model.parameters(), max_iter=max_iter, lr=lr)

    train_ldr = T.utils.data.DataLoader(train_set,
                                        batch_size=len(train_set),
                                        shuffle=False)

    print("Starting L-BFGS training")
    for itr in range(0, 50):
        # The loss for all bathces of the iteration will be accumulated here.
        itr_loss = 0.0

        # Note that the data loader as set up above puts all data in one batch
        # so it's not necessary to make a loop here but for some future
        # flexibility we keep it this way.
        for (_, all_data) in enumerate(train_ldr):
            X = all_data[:, 0:-2]
            Y = all_data[:, -2:-1]
            W = all_data[:, -2:]

            # -------------------------------------------
            # The "closure" is needed because the optimizer should be able to
            # compute loss and gradients repetively for the current model
            # parameters.
            def loss_closure(X=X, Y=Y, W=W):
                opt.zero_grad()
                P = model(X, p_tar)
                loss_val = weighted_loss_func(P, Y, W)
                loss_val.backward()
                return loss_val
            # If we define the function as "def loss_closure(X, Y, W):"
            # we get the flake8 complaint
            # "F841 local variable 'Y' is assigned to but never used", see
            # https://docs.python-guide.org/writing/gotchas
            #         /#late-binding-closures
            # for a general explanation. However, here we will actually
            # not have a problem since we will only use the function
            # within the loop "at the same iteration step" where it is defined.
            # -------------------------------------------

            opt.step(loss_closure)      # get loss, use to update wts

            _ = model(X, p_tar)         # monitor loss
            loss_val = loss_closure()
            itr_loss += loss_val.item()
        # Generate training information message.
        message = "iteration = {: >4d} loss = {: 0.4f}. Weights {} bias {}"
        message = message.format(itr, itr_loss,
                                 list(model.parameters())[0].detach().numpy(),
                                 list(model.parameters())[1].detach().numpy())
        print(message)
        #
    print("Done ")


def estimate_min_max_fusion(train_set):
    # Scores can be normalized into the 0-1 range by
    # (score - min) / (max - min) = score * 1 / (max - min) - min / (max -min)
    # i.e., the affine transform has
    # scale = 1 / (max - min)
    # offset = - min / (max - min) = - min * scale
    # The above is for all scores. For their average, we can combine their
    # offsets into one.
    n_trials, n_sys = train_set.shape
    print("n_trials {}, n_sys {}".format(n_trials, n_sys))
    scale = np.ones(n_sys)
    offset = np.zeros(n_sys)
    for i in range(n_sys):
        mi = np.min(train_set[:, i])
        ma = np.max(train_set[:, i])

        scale[i] = 1 / (ma - mi)
        offset[i] = -mi * scale[i]

    # Make the final transform to be mean instead of sum
    # score_new = (1 / n_sys) *  sum_i [  score[i]*scale[i] +  offset[i] ]
    #   = sum_i [  score[i]*( scale[i] / n_sys) +  ( offset[i] / n_sys ) ]
    scale = scale / n_sys
    offset = offset / n_sys

    # Sum up the offsets into one to fit the standard "affine transform" format
    aff_trans = np.concatenate((scale, np.sum(offset, keepdims=True)))
    print("Estimated affine transform: scales {}, offset {}"
          .format(aff_trans[0:-1], aff_trans[-1]))
    return aff_trans


def main(score_dir, cm_key, model_save_path,
         p_tar=0.5, c_fr=1, c_fa=1, method='LR'):

    # Compute effective prior
    p_eff = p_tar * c_fr / (p_tar * c_fr + (1-p_tar) * c_fa)
    print("p_tar: {}, c_fr: {}, c_fa: {}, p_eff: {}.".format(
            p_tar, c_fr, c_fa, p_eff))

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
        df_tmp = pd.read_csv(
            score_dir + '/' + systems[i] + '/llr.txt', sep='\t')
        df_tmp = df_tmp.rename(
            columns={'cm-score': 'cm-score' + str(i)}).set_index('filename')
        # To consider: Would it better to put the system's name in the column
        #              names? (For clearer debugging / analysis.)
        if i == 0:
            df = df_tmp
        else:
            df = df.join(df_tmp, on='filename')

    # Read in the labels and join to the dataframe
    df_tmp = pd.read_csv(cm_key, sep='\t')
    df = df.join(df_tmp.set_index('filename'), on='filename')

    df.replace(to_replace={'bonafide': 1, 'spoof': 0}, inplace=True)

    # In the end, we want the loss funtion to be the (empirical) expected value
    # of the loss of a trial, i.e.,
    # (p_eff / #tar ) * sum_{i=1}^#{tar} Loss(Xt_i,Yt_i)
    #   + (1-p_eff / #non)*sum_{i=1}^#non Loss(Xn_i,Yn_i)
    # It seems using weights p_eff/n_tar and (1-p_eff)/n_non and later reducing
    # by sum causes numerical instability. Therefores we scale also with
    # n_trials here and later reduce with mean instead of sum. Not sure if this
    # is stable enough in e.g. heavily unbalanced data sets.
    n_trials = len(df)
    n_tar = df['cm-label'].sum()
    n_non = n_trials - n_tar

    print("#trials {}, #tar {}, #non-target {}".format(n_trials, n_tar, n_non))

    df['weight'] = df['cm-label'].replace(
        to_replace={1: n_trials*p_eff/n_tar, 0: n_trials*(1-p_eff)/n_non})

    df = df.to_numpy().astype(np.float32)

    device = T.device("cpu")

    if method == 'LR':
        print("Estimating calibration / fusion parameters with"
              "Logistic regression")
        # Initialize model. TODO: Code should support more models.
        model = LogisticRegression(n_sys=n_sys, p_tar=p_eff).to(device)

        # Train
        train_with_BFGS(model=model, train_set=df)

        # Save model
        T.save(model.state_dict(), model_save_path)

    elif method == 'min_max_avg':
        print("Estimating normaliztion / fusion parameters with"
              "min-max (0-1) normalization")
        # The last two columns are label and weight which are not needed for
        # this method so discard them.
        aff_trans = estimate_min_max_fusion(train_set=df[:, 0:-2])
        np.save(model_save_path, aff_trans, allow_pickle=False)


if __name__ == "__main__":
    fire.Fire(main)
