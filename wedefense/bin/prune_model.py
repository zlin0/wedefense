# Copyright (c) 2025 Jiangyu Han, Junyi Peng
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

import os
import copy
import json
import fire
import torch
from wedefense.frontend import frontend_class_dict
from wedefense.models.get_model import get_model
from wedefense.utils.checkpoint import load_checkpoint
from wedefense.utils.utils import parse_config_or_kwargs
from wedefense.frontend.wav2vec2.model import wav2vec2_model


def extract(config='conf/config.yaml', **kwargs):
    # parse configs first
    configs = parse_config_or_kwargs(config, **kwargs)

    model_path = configs['model_path']
    out_dir = configs.get('out_dir', './pruned_model')
    os.makedirs(out_dir, exist_ok=True)

    # ---------- build & load ------------------------------------------------
    test_conf = copy.deepcopy(configs['dataset_args'])
    # model: frontend (optional) => speaker model

    frontend_type = test_conf.get('frontend', 'fbank')
    if frontend_type != "fbank" and not frontend_type.startswith('lfcc'):
        frontend_args = frontend_type + "_args"
        # frontends besides acoustic features, like s3prl
        print('Initializing frontend model (this could take some time) ...')
        frontend = frontend_class_dict[frontend_type](
            **test_conf[frontend_args], sample_rate=test_conf['resample_rate'])
        configs['model_args']['feat_dim'] = frontend.output_size()
        model = get_model(configs['model'])(**configs['model_args'])
        model.add_module("frontend", frontend)

    print('Load checkpoint ...')
    load_checkpoint(model, model_path)

    print('Start Pruning ...')
    conv_config, use_attention, use_feed_forward, num_heads, remaining_heads, ff_interm_features = model.frontend.prune(
    )

    pruned_config = model.frontend.upstream_config.copy()
    if len(num_heads) == 0:  # for wavlm
        assert len(remaining_heads) > 0
        pruned_config.update({
            "encoder_remaining_heads": remaining_heads,
        })
    else:
        pruned_config.update({
            "encoder_num_heads": num_heads,
        })
    pruned_config.update({
        "extractor_conv_layer_config": conv_config,
        "encoder_use_attention": use_attention,
        "encoder_use_feed_forward": use_feed_forward,
        "encoder_ff_interm_features": ff_interm_features,
        "extractor_prune_conv_channels": False,
        "encoder_prune_attention_heads": False,
        "encoder_prune_attention_layer": False,
        "encoder_prune_feed_forward_intermediate": False,
        "encoder_prune_feed_forward_layer": False,
        "use_layerwise_prune": False
    })

    print('saving pruned model...')
    pruned_out_path = os.path.join(out_dir, 'pytorch_model.bin')
    pruned_out_path_with_backend = os.path.join(out_dir,
                                                'whole_pytorch_model.bin')
    torch.save(
        {
            "state_dict": model.frontend.upstream.state_dict(),
            "config": pruned_config,
        }, pruned_out_path)
    torch.save(model.state_dict(), pruned_out_path_with_backend)
    # pruned_config['wavlm_original_params'] = f'{original_num_params} M'
    print(f"Successfully saved pruned model weights and config to: {out_dir}")

    # ---------- Verify & param count ----------
    ckpt = torch.load(pruned_out_path, map_location="cpu")
    model_verify = wav2vec2_model(**ckpt['config'])

    print(model_verify.load_state_dict(ckpt['state_dict'], strict=False))
    cur_num_params = sum(p.numel() for p in model_verify.parameters()) / 1e6
    print(f'current_num_params: {cur_num_params} M')

    # ---------- Save config as JSON -----------
    json_out_path = os.path.join(out_dir, 'pruned_config.json')
    with open(json_out_path, 'w') as file:
        json.dump(pruned_config, file, indent=4)
    print(f"Successfully saved pruned config to: {json_out_path}")


if __name__ == '__main__':
    fire.Fire(extract)
