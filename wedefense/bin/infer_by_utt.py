#!/usr/bin/env python

# Copyright (c) 2024 Johan Rohdin (rohdin@fit.vutbr.cz)
#               2025 Lin Zhang (partialspoof@gmail.com)
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
wedefense/bin/infer_by_utt.py

The predicted logits and posteriors are generated from the embeddings.
The inference is conducted by iterating through each utterance and
saving the results (logits and posteriors) in ark and scp formats.

"""

import fire
import kaldiio
import numpy as np
import torch
from wedefense.utils.utils import parse_config_or_kwargs
from wedefense.models.projections import get_projection
from scipy.special import softmax


def main(model_path,
         config,
         num_classes,
         embedding_scp_path,
         out_path,
         data_type="raw"):
    # data_type is just used for seting up the projection properly.

    print("model_path {}".format(model_path))
    print("config {}".format(config))
    print("embedding_scp_path {}".format(embedding_scp_path))
    print("out_path {}".format(out_path))

    checkpoint = torch.load(model_path, map_location='cpu')
    configs = parse_config_or_kwargs(config)

    # projection layer
    if (configs['model_args']['embed_dim'] < 0):
        # #if emb_dim <0, we will reduce dim by emb_dim. like -2 will be dim/2
        if 'multireso' in configs[
                'model'] and configs['model_args']['num_scale'] > 0:
            # If we are using multireso structure, dim will reduced by
            # ['embed_dim'] in ['num_scale'] times.
            configs['projection_args']['embed_dim'] = int(
                configs['model_args']['feat_dim'] /
                pow(abs(configs['model_args']['embed_dim']),
                    configs['model_args']['num_scale']))
        else:
            configs['projection_args']['embed_dim'] = int(
                configs['model_args']['feat_dim'] /
                abs(configs['model_args']['embed_dim']))
    else:
        configs['projection_args']['embed_dim'] = configs['model_args'][
            'embed_dim']
    configs['projection_args']['num_class'] = num_classes
    configs['projection_args']['do_lm'] = configs.get('do_lm', False)
    if data_type != 'feat' and configs['dataset_args']['speed_perturb']:
        # diff speed is regarded as diff spk
        configs['projection_args']['num_class'] *= 3
        if configs.get('do_lm', False):
            logger.info(
                'No speed perturb while doing large margin fine-tuning')
            configs['dataset_args']['speed_perturb'] = False
    projection = get_projection(configs['projection_args'])

    # trick
    new_checkpoint = {}
    for k in checkpoint.keys():
        if 'projection.' in k:
            new_checkpoint[k.replace('projection.', '')] = checkpoint[k]
    missing_keys, unexpected_keys = projection.load_state_dict(new_checkpoint,
                                                               strict=False)
    if (len(missing_keys) > 0):
        print("WARNING: {} missing_keys.".format(len(missing_keys)))
    if (len(unexpected_keys) > 0):
        print("WARNING: {} unexpected_keys.".format(len(unexpected_keys)))

    device = torch.device("cpu")
    projection.to(device).eval()

    # Initialize
    utts_lst = []
    embds_lst = []
    with torch.no_grad():
        with kaldiio.WriteHelper('ark,scp:' + out_path + "/logits.ark," + out_path + "/logits.scp") as writer_logit, \
             kaldiio.WriteHelper('ark,scp:' + out_path + "/posteriors.ark," + out_path + "/posteriors.scp") as writer_post:

            for utt, emb in kaldiio.load_scp_sequential(embedding_scp_path):
                utts_lst.append(utt)
                embds_lst.append(emb)

                output = projection(torch.from_numpy(emb.copy()),
                                    torch.from_numpy(np.zeros(emb.shape[0])))

                if isinstance(output, tuple):
                    # some projection layers return output and loss
                    output = output[0].detach().numpy()
                else:
                    output = output.detach().numpy()

                writer_logit(utt, output)
                writer_post(utt, softmax(output))

    embds = np.vstack(embds_lst)
    print(embds.shape)

    print(f"Save logits and posteriors to {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
