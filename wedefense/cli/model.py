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
import yaml

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from wedefense.frontend import frontend_class_dict
from wedefense.models.projections import get_projection
from wedefense.models.get_model import get_model
from wedefense.utils.checkpoint import load_checkpoint
from wedefense.dataset.dataset_utils import apply_cmvn
from wedefense.cli.hub import Hub


class Model:

    def __init__(self,
                 model_dir: str,
                 config_name: str = "config.yaml",
                 model_name: str = "avg_model.pt"):

        config_path = os.path.join(model_dir, config_name)
        model_path = os.path.join(model_dir, model_name)
        with open(config_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)

        self.model = self.init_model(configs, model_path)
        self.resample_rate = 16000
        self.device = torch.device('cpu')
        self.wavform_normalize = True

        self.cmvn = configs['dataset_args'].get('cmvn', True)
        self.cmvn_args = configs['dataset_args'].get('cmvn_args', {})

    def init_model(self, configs, model_path):
        self.frontend_type = configs['dataset_args'].get(
            'frontend', 'fbank')  #TODO support other features
        if self.frontend_type != "fbank" and not self.frontend_type.startswith(
                'lfcc'):
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

    def compute_embeds(self, audio_path: str, detection=False):
        wavform, sample_rate = torchaudio.load(audio_path,
                                               normalize=self.wavform_normalize)
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
        else:
            raise NotImplementedError("Unsupported frontend type: {}".format(
                self.frontend_type))

        if self.cmvn:
            features = apply_cmvn(features, **self.cmvn_args)

        if not detection and hasattr(self.model, 'get_frame_emb'):
            outputs = self.model.get_frame_emb(features)
        else:
            outputs = self.model(features)  # (B,T,D)

        embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
        return embeds

    def logits_to_rttm(self,
                       logits,
                       utt="test_audio",
                       score_reso=20,
                       bonafide_idx=0):
        rttm = []
        frame_shift = score_reso / 1000.0  # Convert ms to seconds
        preds = logits.argmax(axis=1)  # Frame-level predictions (0 or 1)
        if len(preds) == 0:
            return []
        start_idx = 0
        current_label = preds[0]
        for idx in range(1, len(preds)):
            if preds[idx] != current_label:
                start_time = start_idx * frame_shift
                duration = (idx - start_idx) * frame_shift
                label = 'bonafide' if current_label == bonafide_idx else 'spoof'
                rttm.append(
                    f"Wedefense {utt} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {label} <NA> <NA>"
                )
                start_idx = idx
                current_label = preds[idx]

        start_time = start_idx * frame_shift
        duration = (len(preds) - start_idx) * frame_shift
        label = 'bonafide' if current_label == bonafide_idx else 'spoof'
        rttm.append(
            f"Wedefense {utt} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {label} <NA> <NA>"
        )
        return rttm

    def detection_probs(self, audio_path: str):
        embeds = self.compute_embeds(audio_path, detection=True)
        outputs = self.model.projection(embeds)
        outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        outputs = torch.nn.functional.softmax(outputs, dim=-1)  # [B, 2]
        return outputs

    def detection(self, audio_path: str):
        outputs = self.detection_probs(audio_path).squeeze(0)  # [2]
        if outputs[0] > outputs[1]:
            print(f"The audio is {outputs[0]*100:.2f}% bonafide")
        else:
            print(f"The audio is {outputs[1]*100:.2f}% spoof")

    def localization_logits(self, audio_path: str):
        embeds = self.compute_embeds(audio_path)
        outputs = self.model.projection(embeds.squeeze(0))
        outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        return outputs

    def localization(self, audio_path: str, rttm_file: str):
        outputs = self.localization_logits(audio_path)
        rttm = self.logits_to_rttm(outputs,
                                   os.path.basename(audio_path).split('.')[0])
        with open(rttm_file, 'w', encoding='utf-8') as fout:
            fout.write("\n".join(rttm))
        return outputs


def load_model(model_id: str = None, model_dir: str = None):
    if model_dir is None:
        model_dir = Hub.get_model(model_id)

    return Model(model_dir)
