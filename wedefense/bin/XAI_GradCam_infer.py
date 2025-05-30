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

import copy
import os
import pickle
# import kaldiio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch import nn

from wedefense.dataset.dataset import Dataset
from wedefense.dataset.dataset_utils import apply_cmvn
from wedefense.frontend import *
from wedefense.models.get_model import get_model
from wedefense.utils.checkpoint import load_checkpoint
# from wedefense.utils.utils import parse_config_or_kwargs, validate_path

import fire
# import kaldiio
import numpy as np
from wedefense.utils.utils import parse_config_or_kwargs
from wedefense.models.projections import get_projection
import os.path
# from scipy.special import softmax

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from wedefense.dataset.dataset_utils import apply_cmvn

class FullModel(nn.Module):
    def __init__(self, model, projection, test_conf):
        super().__init__()
        self.frontend = model.frontend
        self.encoder = model
        self.projection = projection
        self.test_conf = test_conf

    def forward(self, wavs):
        wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(wavs.shape[0]).to(wavs.device)

        features, _ = self.frontend(wavs, wavs_len)               # Step 1: Front-end
        features = apply_cmvn(features, **self.test_conf.get('cmvn_args', {}))
        embeddings = self.encoder(features)                       # Step 2: Encoder (e.g., ResNet, TDNN)

        # Step 3: Add dummy labels (all zeros)
        dummy_labels = torch.zeros(embeddings.size(0), dtype=torch.long).to(wavs.device)

        # Step 4: Projection (with two inputs)
        out = self.projection(embeddings, dummy_labels)

        # If projection returns (output, loss), extract output
        if isinstance(out, tuple):
            out = out[0]

        return out

    
def main(config='conf/config.yaml', **kwargs):
    # parse configs first
    configs = parse_config_or_kwargs(config, **kwargs)
    print('configs', configs)

    model_path = configs['model_path']
    xai_scores_path = configs['xai_scores_path']
    num_classes = configs['num_classes']
    data_type = configs['data_type']
    batch_size = configs['batch_size']
    assert batch_size == 1
    num_workers = configs['num_workers']
    # embedding_scp_path = configs['embedding_scp_path']
    # out_path = configs['out_path']
    # print(configs)
    # print("model_path {}".format( model_path  ) )
    # print("config {}".format( config ) )
    # print("embedding_scp_path {}".format( embedding_scp_path ) )
    # print("out_path {}".format( out_path ) )

    # parse configs first
    # configs = parse_config_or_kwargs(config, **kwargs)
    # configs = config    
    # model_path = configs['model_path']
    # embed_ark = configs['embed_ark']
    # batch_size = configs.get('batch_size', 1)
    # num_workers = configs.get('num_workers', 1)

    # Since the input length is not fixed, we set the built-in cudnn
    # auto-tuner to False
    torch.backends.cudnn.benchmark = False

    test_conf = copy.deepcopy(configs['dataset_args'])
    # model: frontend (optional) => speaker model
    model = get_model(configs['model'])(**configs['model_args'])
    frontend_type = test_conf.get('frontend', 'fbank')
    if frontend_type != "fbank" and not frontend_type.startswith('lfcc'):
        frontend_args = frontend_type + "_args"
        # frontends besides acoustic features, like s3prl
        print('Initializing frontend model (this could take some time) ...')
        frontend = frontend_class_dict[frontend_type](
            **test_conf[frontend_args], sample_rate=test_conf['resample_rate'])
        model.add_module("frontend", frontend)
    print('Loading checkpoint ...')
    load_checkpoint(model, model_path)
    device = torch.device("cuda")
    model.to(device).eval()


    
    # utt  = []
    # embd = []
    # for k, v in kaldiio.load_scp_sequential( embedding_scp_path ):
    #     utt.append( k )
    #     embd.append( v )
    # embd = np.vstack( embd )    
 

    checkpoint = torch.load(model_path, map_location='cpu')

    # configs = parse_config_or_kwargs(config)

    # projection layer
    if(configs['model_args']['embed_dim'] < 0): #TODO check
        # #if emb_dim <0, we will reduce dim by emb_dim. like -2 will be dim/2
        if 'multireso' in configs['model'] and configs['model_args']['num_scale'] > 0:
            # If we are using multireso structure, dim will reduced by 
            #['embed_dim'] in ['num_scale'] times.
            configs['projection_args']['embed_dim'] = int(
                    configs['model_args']['feat_dim'] / 
                    pow(abs(configs['model_args']['embed_dim']), 
                        configs['model_args']['num_scale'])
                    )
        else:
            configs['projection_args']['embed_dim'] = int(
                    configs['model_args']['feat_dim'] / 
                    abs(configs['model_args']['embed_dim'])
                    )
    else:
        configs['projection_args']['embed_dim'] = configs['model_args'][
            'embed_dim']
    configs['projection_args']['num_class'] = num_classes
    configs['projection_args']['do_lm'] = configs.get('do_lm', False)
    if data_type != 'feat' and configs['dataset_args'][
            'speed_perturb']:
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
            new_checkpoint[k.replace('projection.','')] = checkpoint[k]
    missing_keys, unexpected_keys = projection.load_state_dict(new_checkpoint,
                                                               strict=False)
    if (len(missing_keys)>0):
        print( "WARNING: {} missing_keys.".format( len(missing_keys)  ))
    if (len(unexpected_keys)>0):
        print( "WARNING: {} unexpected_keys.".format( len(unexpected_keys)  ))

    # device = torch.device("cpu")
    projection.to(device).eval()

    full_model = FullModel(model, projection, test_conf).to(device).eval()
    del model 
    del projection

    dataset = Dataset(configs['data_type'],
                      configs['data_list'],
                      test_conf,
                      spk2id_dict={},
                      whole_utt=(batch_size == 1),
                      reverb_lmdb_file=configs.get('reverb_data', None),
                      noise_lmdb_file=configs.get('noise_data', None),
                      repeat_dataset=False)
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            prefetch_factor=4)
    
    # wavs = torch.rand(1, 1, 160000).to(device)     # (B, 1, W)
    # wavs = wavs.squeeze(1).float().to(device)      # (B, W)

    # with torch.no_grad():
    #     output = full_model(wavs, test_conf)
    # print('output', output)
    # x = input('?')

    save_list = []  
    save_dir = xai_scores_path  

    targets = [ClassifierOutputTarget(1)]
    target_layer = [full_model.encoder.fc]
    cam = GradCAM(model=full_model, target_layers=target_layer)

    # with torch.no_grad():

        # with kaldiio.WriteHelper('ark,scp:' + embed_ark + "," +
        #                         embed_scp) as writer:
    for _, batch in tqdm(enumerate(dataloader)):
        utt = batch['key'][0] # during inference, batch size shall be 1
        if frontend_type == 'fbank' or frontend_type.startswith('lfcc'):
            features = batch['feat']
            features = features.float().to(device)  # (B,T,F)
        else:  # 's3prl'
            wavs = batch['wav']  # (B,1,W)
            wavs = wavs.squeeze(1).float().to(device)  # (B,W)
            # wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                # wavs.shape[0]).to(device)  # (B).
            

        out_cam = cam(input_tensor=wavs, targets=targets)
        save_list.append([[utt], out_cam.squeeze(0).tolist()])
        # print('utt', utt, 'out_cam.shape', out_cam.shape)

            # output = full_model(wavs, test_conf)

            # print('output', output)
            # apply cmvn
            # if test_conf.get('cmvn', True):
            #     features = apply_cmvn(features,
            #                         **test_conf.get('cmvn_args', {}))
            # # spec augmentation
            # if test_conf.get('spec_aug', False):
            #     features = spec_aug(features, **test_conf['spec_aug_args'])

            # Forward through model
            # outputs = model(features)  # embed or (embed_a, embed_b)
            # embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            # embeds = embeds.cpu().detach().numpy()  # (B,F)

            # for i, utt in enumerate(utts):
            #     embed = embeds[i]
            #     writer(utt, embed)
    with open(xai_scores_path, 'wb') as f:
        pickle.dump(save_list, f)
    print(f"Saved XAI Grad-CAM scores to {save_path}")

    # wavs = torch.rand(1, 1, 160000).to(device)     # (B, 1, W)
    # wavs = wavs.squeeze(1).float().to(device)      # (B, W)

    # with torch.no_grad():
    #     output = full_model(wavs, test_conf)


    # Need to add something here to set margin to zero in case of margin based losses
    # Also a dummy label need to be provided below

    # with torch.no_grad():
    #     output = projection(torch.from_numpy(embd), torch.from_numpy(np.zeros(embd.shape[0])))
    #     if isinstance(output, tuple):
    #         # some projection layers return output and loss
    #         output = output[0].detach().numpy()
    #     else:
    #         output = output.detach().numpy()
    
    # print(output.shape)
    # #print(output)

    # # out_path = os.path.dirname(embedding_scp_path)
    # print(out_path)
    # with kaldiio.WriteHelper('ark,scp:' + out_path + "/logits.ark," + out_path + "/logits.scp") as writer:
    #     for i, utt in enumerate(utt):
    #         writer(utt, output[i])

    # with kaldiio.WriteHelper('ark,scp:' + out_path + "/posteriors.ark," + out_path + "/posteriors.scp") as writer:
    #     for i, utt in enumerate(utt):
    #         writer(utt, softmax(output[i]))

if __name__ == "__main__":
    fire.Fire(main)

