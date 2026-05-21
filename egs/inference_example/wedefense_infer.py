#!/usr/bin/env python

import copy
import numpy as np
import soundfile as sf
import torch

from io import BytesIO

from wedefense.dataset.dataset_utils import apply_cmvn, spec_aug
from wedefense.frontend import frontend_class_dict
from wedefense.models.get_model import get_model
from wedefense.models.projections import get_projection
from wedefense.utils.checkpoint import load_checkpoint
from wedefense.utils.utils import parse_config_or_kwargs


"""

Inference example for single-input MHFA model:

model=https://huggingface.co/JYP2024/Wedefense_ASV2025_WavLM_Base_Pruning
recipe=https://github.com/zlin0/wedefense/tree/main/egs/detection/asvspoof5/v15_ssl_mhfa_pruning

"""


class TtsDetector:
    def __init__(self):
        config = "/mnt/matylda5/iveselyk/VALKA-AI/TTS_DETECTOR_WEDEFENSE/models/config.yaml"
        #config = "/mnt/matylda5/iveselyk/VALKA-AI/TTS_DETECTOR_WEDEFENSE/models/config_pruned.yaml"  # missing `exp/v4/MHFA_wavlm_s10/pruned_model/pytorch_model.bin`

        #configs = parse_config_or_kwargs(config, {})  # empty **kwargs -> {}
        configs = parse_config_or_kwargs(config)  # empty **kwargs -> {}

        model_path = "/mnt/matylda5/iveselyk/VALKA-AI/TTS_DETECTOR_WEDEFENSE/models/models/avg_model.pt"
        #model_path = "/mnt/matylda5/iveselyk/VALKA-AI/TTS_DETECTOR_WEDEFENSE/models/pruned_model/whole_pytorch_model.bin"
        #model_path = "/mnt/matylda5/iveselyk/VALKA-AI/TTS_DETECTOR_WEDEFENSE/models/pruned_model/pytorch_model.bin"

        device = torch.device('cpu')

        # -----

        # [extract.py]

        model = get_model(configs['model'])(**configs['model_args'])

        test_conf = copy.deepcopy(configs['dataset_args'])

        frontend_type = test_conf.get('frontend', 'fbank')
        if frontend_type != "fbank" and not frontend_type.startswith('lfcc'):
            frontend_args = frontend_type + "_args"
            frontend = frontend_class_dict[frontend_type](
                **test_conf[frontend_args],
                sample_rate=test_conf['resample_rate'],
            )
            model.add_module("frontend", frontend)

        load_checkpoint(model, model_path)

        model.to(device).eval()

        # [infer.py]
        checkpoint = torch.load(model_path, map_location='cpu')

        projection = get_projection(configs['projection_args'])

        # trick (assign projection parameters)
        new_checkpoint = {}
        for k in checkpoint.keys():
            if 'projection.' in k:
                new_checkpoint[k.replace('projection.', '')] = checkpoint[k]

        missing_keys, unexpected_keys = projection.load_state_dict(
            new_checkpoint,
            strict=False,
        )

        if (len(missing_keys) > 0):
            print(f"WARNING: {len(missing_keys)} missing_keys: {missing_keys}")
        if (len(unexpected_keys) > 0):
            print(f"WARNING: {len(unexpected_keys)} unexpected_keys: {unexpected_keys}")

        projection.to(device).eval()

        # store
        self.device = device
        self.model = model
        self.test_conf = test_conf
        self.projection = projection

    def inference(self, audio_bytes):
        """
        Process audio file (no batching), get tts-detection score.
        """
        # get pointers
        device = self.device
        model = self.model
        test_conf = self.test_conf
        projection = self.projection

        # import audio
        audio, sampling_freq = sf.read(BytesIO(audio_bytes))
        assert sampling_freq == 16000
        # TODO: resample to 16khz ?

        # [wedefense/bin/extract.py]
        wavs = torch.tensor(audio).float().to(device)  # (W)
        wavs = torch.unsqueeze(wavs, 0)  # (B=1,W)
        wavs_len = torch.LongTensor([len(audio)]).to(device)  # (B=1)

        # frontend
        features, _ = model.frontend(wavs, wavs_len)  # heavy_lifting

        # apply cmvn
        if test_conf.get('cmvn', True):
            features = apply_cmvn(features,
                                  **test_conf.get('cmvn_args', {}))

        # spec augmentation
        if test_conf.get('spec_aug', False):
            features = spec_aug(features, **test_conf['spec_aug_args'])

        # Forward through model
        outputs = model(features)  # embed (B=1,F=256)
        embeds = outputs[-1] if isinstance(outputs, tuple) else outputs

        # [wedefense/bin/infer.py]
        with torch.no_grad():
            output = projection(
                embeds,
                torch.zeros(embeds.shape[0]),
            )
            if isinstance(output, tuple):
                # some projection layers return output and loss
                output = output[0]

        # [wedefense/bin/logits_to_llr.py]
        # skipping this, we don't know the prior, instead
        # we'll produce logit-posteriors calibrate by LogisticRegression, if needed...

        logit = output
        log_post = torch.softmax(logit, axis=1).logit()[0][0]

        return float(log_post.cpu())  # log_posterior


def main():

    # load a 16khz audio
    audio_file = "/mnt/matylda5/iveselyk/VALKA-AI/example_data/test_file_16khz_EN_tj4gZ4d_Q7c_W000115.wav"
    with open(audio_file, 'rb') as fd:
        audio_bytes = fd.read()

    detector = TtsDetector()

    ret = detector.inference(audio_bytes)

    print(ret)


if __name__ == "__main__":
    main()
