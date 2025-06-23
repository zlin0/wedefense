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
logistic_regression.py
A logistic regression model for calibration and/or fusion of binary
classifiers.
Reference:
* "Fusion of Heterogeneous Speaker Recognition Systems in the STBU Submission
   for the NIST Speaker Recognition Evaluation 2006" By Niko Brummer et al.
   Published in: IEEE Transactions on Audio, Speech, and Language Processing
   (Volume: 15, Issue: 7, September 2007)
"""

import torch as T


class LogisticRegression(T.nn.Module):

    def __init__(self, n_sys, p_tar=0.5):
        super(LogisticRegression, self).__init__()
        # The specified prior can be overidden when calling forward.
        self.p_tar = T.tensor(p_tar, requires_grad=False)
        self.trans = T.nn.Linear(n_sys, 1)

        # If scores are calibrated and independent, the weights should be
        # like this (the training will fix that this is not the case):
        T.nn.init.ones_(self.trans.weight)
        T.nn.init.zeros_(self.trans.bias)

    def forward(self, x, p_tar=-1):
        # We have
        #
        #  log posterior odds
        #    = log( P(tar|X) / P(non|X) )
        #    = log( P(X|tar) / P(X|non) ) + log(P(tar)/(P(non)))
        #    = LLR - tau
        # where tau = -log(P(tar)/(P(non))) is the decision threshold for equal
        # costs. (We should accept if P(tar|X) > P(non|X) <=> LLR > tau. )
        #
        # By adding -tau = log(p_tar/(1-p_tar)) to the scores before sending
        # them to the loss function, we enforce them to be LLRs instead of
        # log posterior odds. (Recall that our loss function enourages the its
        # input to be log posterior odds.)

        if p_tar == -1:
            tau = - T.special.logit(self.p_tar)
        else:
            tau = - T.special.logit(T.tensor(p_tar, requires_grad=False))
        s = self.trans(x) - tau
        return s
