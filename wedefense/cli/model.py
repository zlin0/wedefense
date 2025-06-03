# Copyright (c) 2025 Chengdong Liang (liangchengdongd@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import yaml

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from wedefense.frontend import frontend_class_dict
from wedefense.models.projections import get_projection
from wedefense.models.get_model import get_model
from wedefense.utils.checkpoint import load_checkpoint
from wedefense.dataset.dataset_utils import apply_cmvn


class Model:
    def __init__(self, model_dir: str,
                 config_name: str="config.yaml",
                 model_name: str="avg_model.pt"):

        config_path = os.path.join(model_dir, config_name)
        model_path = os.path.join(model_dir, model_name)
        with open(config_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        print(configs)

        self.model = self.init_model(configs, model_path)
        self.resample_rate = 16000
        self.device = torch.device('cpu')
        self.wavform_normalize = True

        self.cmvn = configs['dataset_args'].get('cmvn', True)
        self.cmvn_args = configs['dataset_args'].get('cmvn_args', {})


    def init_model(self, configs, model_path):
        self.frontend_type = configs['dataset_args'].get('frontend', 'fbank') #TODO support other features
        if self.frontend_type != "fbank" and not self.frontend_type.startswith('lfcc'):
            frontend_args = self.frontend_type + "_args"
            frontend = frontend_class_dict[self.frontend_type](
                **configs['dataset_args'][frontend_args],
                sample_rate=configs['dataset_args']['resample_rate'])
            configs['model_args']['feat_dim'] = frontend.output_size()
            model = get_model(configs['model'])(**configs['model_args'])
            model.add_module("frontend", frontend)
        else:
            model = get_model(configs['model'])(**configs['model_args'])

        projection = get_projection(configs["projection_args"])
        model.add_module("projection", projection)
        load_checkpoint(model, model_path)
        model.eval()
        return model


    def set_resample_rate(self, resample_rate: int):
        self.resample_rate = resample_rate

    def set_device(self, device: str):
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

    def set_wavform_normalize(self, normalize: bool):
        self.wavform_normalize = normalize

    def compute_fbank(self,
                      wavform,
                      sample_rate=16000,
                      num_mel_bins=80,
                      frame_length=25,
                      frame_shift=10,
                      cmn=True):
        feat = kaldi.fbank(wavform,
                           num_mel_bins=num_mel_bins,
                           frame_length=frame_length,
                           frame_shift=frame_shift,
                           sample_frequency=sample_rate,
                           window_type='hamming')
        if cmn:
            feat = feat - torch.mean(feat, 0)
        return feat

    def compute_embeds(self, audio_path: str):
        wavform, sample_rate = torchaudio.load(audio_path,
                                               normalize=self.wavform_normalize)
        print(wavform.shape)
        if sample_rate != self.resample_rate:
            wavform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.resample_rate)(wavform)

        if self.frontend_type == "fbank":
            features = self.compute_fbank(wavform)
            features = features.unsqueeze(0).to(self.device)
        elif self.frontend_type == "s3prl":
            wavform = wavform.to(self.device)  # (B, W)
            wavform_len = torch.LongTensor([wavform.shape[1]]).repeat(
                wavform.shape[0]).to(self.device)  # (B)
            features, _ = self.model.frontend(wavform, wavform_len)
            print(wavform_len)
        else:
            raise NotImplementedError("Unsupported frontend type: {}".format(
                self.frontend_type))

        if self.cmvn:
            features = apply_cmvn(features, **self.cmvn_args)

        outputs = self.model(features)
        embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
        return embeds

    def detection(self, audio_path: str):
        embeds = self.compute_embeds(audio_path)
        outputs = self.model.projection(embeds)
        outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        print(outputs)
        return outputs



    def localization(self, audio_path: str):
        embeds = self.compute_embeds(audio_path)
        outputs = self.model.projection(embeds.squeeze(0))
        outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        print(outputs)
        return outputs

def load_model(model_id: str=None, model_dir: str=None):
    if model_dir is None:
        model_dir = Hub.get_model(model_id)

    return Model(model_dir)

