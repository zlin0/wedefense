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

import kaldiio
import numpy as np
import os

import umap
import matplotlib
import matplotlib.pyplot as plt


def get_utt2embs(emb_scp):
    utt2embs = {}
    for utt, emb in kaldiio.load_scp_sequential(emb_scp):
        utt2embs[utt] = emb
    return utt2embs


def draw_emb_fig(ax, emb_scp, utt2lab, label2num, num2label, redudim_name=""):
    """
    ax: The matplotlib Axes object to draw on
    emb_scp:
    """

    print("start to process embedding from " + emb_scp)
    utt2embs = get_utt2embs(emb_scp)
    os.makedirs("embs", exist_ok=True)

    all_emb, all_lab = [], []
    for utt, emb in utt2embs.items():
        if utt not in utt2lab:
            continue
        all_emb.append(emb)
        all_lab.append(utt2lab[utt])

    all_emb_ary = np.vstack(all_emb)
    all_labnum = [int(label2num[i]) for i in all_lab]

    # UMAP
    if len(redudim_name) > 0:
        redudim_file = os.path.join(
            "embs",
            redudim_name if redudim_name.endswith(".npy") else redudim_name +
            ".npy")
    else:
        redudim_file = os.path.join(
            "embs", f"{emb_scp.split('/')[-4]}_{emb_scp.split('/')[-2]}.npy")

    if os.path.exists(redudim_file):
        redu_embedding = np.load(redudim_file)
    else:
        reducer = umap.UMAP(n_components=2,
                            n_neighbors=15,
                            min_dist=0.1,
                            metric='cosine')
        redu_embedding = reducer.fit_transform(all_emb_ary)
        np.save(redudim_file, redu_embedding)

    norm = plt.Normalize(vmin=min(all_labnum), vmax=max(all_labnum))
    cmap = matplotlib.cm.get_cmap('tab20')

    ax.scatter(redu_embedding[:, 0],
               redu_embedding[:, 1],
               c=all_labnum,
               cmap=cmap,
               norm=norm,
               s=0.2,
               alpha=0.3)

    for tag in np.unique(all_labnum):
        mean_data = redu_embedding[np.array(all_labnum) == tag].mean(axis=0)
        ax.text(mean_data[0],
                mean_data[1],
                num2label[str(tag)],
                fontsize=10,
                color=cmap(norm(tag)),
                bbox=dict(facecolor='white', alpha=0.7))

    ax.set_xticks([])
    ax.set_yticks([])
