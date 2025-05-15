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
import torch
from wedefense.utils.utils import parse_config_or_kwargs
from wedefense.models.projections import get_projection
import os.path
from scipy.special import softmax

def main(model_path, config, num_classes, embedding_scp_path, out_path, data_type="raw"):
    # data_type is just used for seting up the projection properly.
    
    print("model_path {}".format( model_path  ) )
    print("config {}".format( config ) )
    print("embedding_scp_path {}".format( embedding_scp_path ) )
    print("out_path {}".format( out_path ) )

    
    utt  = []
    embd = []
    for k, v in kaldiio.load_scp_sequential( embedding_scp_path ):
        utt.append( k )
        embd.append( v )
    embd = np.vstack( embd )    
    print(embd.shape)    

    checkpoint = torch.load(model_path, map_location='cpu')
    

    configs = parse_config_or_kwargs(config)

    # projection layer
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

    device = torch.device("cpu")
    projection.to(device).eval()

    # Need to add something here to set margin to zero in case of margin based losses
    # Also a dummy label need to be provided below

    with torch.no_grad():
        output = projection(torch.from_numpy(embd), torch.from_numpy(np.zeros(embd.shape[0]))).detach().numpy()
    
    print(output.shape)
    #print(output)

    # out_path = os.path.dirname(embedding_scp_path)
    print(out_path)
    with kaldiio.WriteHelper('ark,scp:' + out_path + "/logits.ark," + out_path + "/logits.scp") as writer:
        for i, utt in enumerate(utt):
            writer(utt, output[i])

    with kaldiio.WriteHelper('ark,scp:' + out_path + "/posteriors.ark," + out_path + "/posteriors.scp") as writer:
        for i, utt in enumerate(utt):
            writer(utt, softmax(output[i]))

if __name__ == "__main__":
    fire.Fire(main)

