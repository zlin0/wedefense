#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 

import os
import sys
import numpy as np

import torchaudio
from torchaudio.io import CodecConfig, AudioEffector

SUPPORTED_CODEC_FOR_AUGMENTATION = ['mp3', 'ogg-vorbis', 'ogg-opus', 'g722', 'mu-law', 'pcm16']

SUPPORTED_CODEC_CONFIG = {'mp3': [8000, 16000, 32000, 64000, 92000, 128000],
                          'ogg-vorbis': [8000, 16000, 32000, 64000, 92000, 128000],
                          'ogg-opus': [8000, 16000, 32000, 64000, 92000, 128000],
                          'g722': [48000, 56000, 64000],
                          'mu-law': [-1],
                          'pcm16': [-1]}

def codec_apply(wav_in, sr, format, bitrate=-1, q_factor=None):
    """ Apply codec to input waveform

        Args:
            wav_in (tensor): waveform in shape [channel, time]
            sr (int): sampling rate
            format (str): name of the format
            bitrate (int): expected bitrate
            q_factor (float): q factor of codec
        Return:
            waveform_coded (tensor): waveform in the same shape as input
    """
    
    assert wav_in.ndim == 2, "Codec_apply: input waveform be in shape [channel, time]"
    assert wav_in.shape[0] == 1 or wav_in.shape[1] == 1, "Codec_apply: not single channel wav"

    # change to [time, channel] format
    wav_ = wav_in.T if wav_in.shape[0] == 1 else wav_in
        
    if format != 'mu-law' and format != 'pcm16':
        assert (q_factor is None and bitrate >= 0) \
            or (q_factor is not None and bitrate == -1), \
            "Codec_apply: please set either bitrate or q_factor"

    codec_config = CodecConfig(bit_rate=bitrate, qscale=q_factor)
    
    if format == 'mp3':
        codec = AudioEffector(format=format, codec_config=codec_config)
    elif format == "ogg-vorbis":
        codec = AudioEffector(format="ogg", encoder="vorbis", codec_config=codec_config)
    elif format == "ogg-opus":
        codec = AudioEffector(format="ogg", encoder="opus", codec_config=codec_config)
    elif format == "g722":
        codec = AudioEffector(format="wav", encoder="g722", codec_config=codec_config)
    elif format == "speex":
        codec = AudioEffector(format="ogg", encoder="libspeex", codec_config=codec_config)
    elif format == "gsm":
        codec = AudioEffector(format="gsm", encoder="libgsm", codec_config=codec_config)
    elif format == "g726":
        codec = AudioEffector(format="g726", encoder="g726", codec_config=codec_config)
    elif format == "mu-law":
        codec = AudioEffector(format="wav", encoder="pcm_mulaw")
    elif format == "pcm16":
        codec = AudioEffector(format="wav", encoder="pcm_s16le")
    else:
        raise ValueError(f"Format '{format}' not supported")        
        
    waveform_coded = codec.apply(wav_, sr)

    if wav_in.shape[0] == 1:
        return waveform_coded.T
    else:
        return waveform_coded


    
