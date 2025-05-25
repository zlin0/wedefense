# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2022 Hongji Wang (jijijiang77@gmail.com)
#               2023 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2025 Lin Zhang (partialspoof@gmail.com)
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

import io
import kaldiio
import json
import logging
import random
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import numpy as np
from scipy import signal
from scipy.io import wavfile
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

import wedefense.dataset.augmentation.rawboost_util as rawboost_util 
from wedefense.utils.diarization.rttm_tool import get_rttm, rttm2vadvec


def label_to_id_timestamps(data, label2id):
    """ Parse spk id

        Args:
            data: Iterable[{key, wav/feat, spk}],
                 'spk': [<str: label>, <float: st>, <float: end>]
            spk2id: Dict[str, int] e.g.:{'bonafide': 1, 'spoof': 0}

        Returns:
            Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert 'spk' in sample
        one_rttm_id = []
        # check whether it is in rttm fotmat
        for seg in sample['spk']:
            label_str, st, et = seg
            label_id = label2id[label_str]
            if label_str in label2id:
                label = label2id[label_str]
            else:
                label = -1
            one_rttm_id.append([label_id, st, et])

        sample['label'] = one_rttm_id
        yield sample


def chunk_label_timestamps(label_timestamps, start, end):
    """ Chunk segments to a specified time interval.
    Args:
        label_timestamps (List[List[int, float, float]]): one utterance's RTTM segments
                         after mapping label to int.        
        start (float): start time of chunk, in second
        end (float): end time of chunk, in second
    Returns:
        List[List[int, float, float]]: chunked segments, time-shifted so new start is 0.0
    """
    new_label_timestamps = []
    assert(start <= end)

    for lab, st, et in label_timestamps:
        # check if segment overlaped with [st, et]
        if(et <= start or st >= end):
            continue #no overlap
        new_st = round(max(st, start) - start, 8)
        new_et = round(min(et, end) - start, 8) # To avoid floating-point precision error in python
        new_label_timestamps.append([lab, new_st, new_et])

    return new_label_timestamps

def pad_label_timestamps(label_timestamps, chunk_len):
    """ Pad segments
    Args:
        label_timestamps (List[List[int, float, float]]): one utterance's RTTM segments
                         after mapping label to int.        
        chunk_len (float): the desired duration. in second, same as in rttm.
    Returns:
        List[List[int, float, float]]: chunked segments, time-shifted so new start is 0.0
    """
    assert label_timestamps[0][1] == 0.0 # Assume the label_timestamps starts from 0.0
                                         # And every secons are labeled.
    duration = label_timestamps[-1][2] # assume the last end is total duration
    repeat_label_timestamps = label_timestamps.copy()

    while(duration < chunk_len):
        for lab, st, et in label_timestamps:
            new_st = st + duration
            new_et = et + duration
            if(new_st >= chunk_len):
                break
            repeat_label_timestamps.append([lab, new_st, min(new_et, chunk_len)])
            duration += (min(new_et, chunk_len) - new_st)

    return repeat_label_timestamps

def get_random_chunk_timestamps(data, label, chunk_len_sp, sample_rate=16000):
    """ Get random chunk, support labels in timestamps.

        Args:
            data: torch.Tensor (random len)
            label: [<str: label>, <float: st>, <float: end>]
            chunk_len_sp: chunk length in sampling point.

        Returns:
            torch.Tensor (exactly chunk_len_sp)
    """
    if chunk_len_sp <= 100:
        raise ValueError("chunk_len_sp must be positive, and is sample point")
    data_len = len(data)
    data_shape = data.shape
    # random chunk
    if data_len >= chunk_len_sp:
        chunk_start = random.randint(0, data_len - chunk_len_sp) #samplepoint
        data = data[chunk_start:chunk_start + chunk_len_sp]
        new_label = chunk_label_timestamps(label, float(chunk_start/sample_rate), 
                                           float((chunk_start+chunk_len_sp)/sample_rate))
        # re-clone the data to avoid memory leakage
        if type(data) == torch.Tensor:
            data = data.clone()
        else:  # np.array
            data = data.copy()
    else:
        # padding
        repeat_factor = chunk_len_sp // data_len + 1
        repeat_shape = repeat_factor if len(data_shape) == 1 else (
            repeat_factor, 1)
        if type(data) == torch.Tensor:
            data = data.repeat(repeat_shape)
        else:  # np.array
            data = np.tile(data, repeat_shape)
        data = data[:chunk_len_sp]
        new_label = pad_label_timestamps(label, float(chunk_len_sp/sample_rate))

    return data, new_label

def timestamps_to_labelvec(data, shift_sec, label2id, reco2dur):
    """ Replace 'spk' segment field with frame-level label vector.
    Args:
        samples (List[Dict]): each sample has 'spk', 'wav'
        shift_sec (float): frame shift
        spk2id (Dict[str, int]): label to id
        reco2dur (Dict[str, float]): utterance duration
    Yields:
        sample['label'] replaced by frame-level label vector
        #note that sample['spk'] has str as label, ['label'] has number as id
        #TODO to be consistent.
    """    

    for sample in data:
        assert 'label' in sample
        assert 'key' in sample
        label = sample['spk']    
        # dur = reco2dur[sample['key']], may chunked, so need to use updated duration.
        # TODO duration is calculated too many times, save duration info to data using processor.pt 
        # Lin 20250524: we need to transfer dur in case rttm doesn't cover all durations.
        # In current case, we assigned all durations, so use -1
        # If needed, uncomment: 
        # dur = float(len(sample['wav'])/sample['sample_rate'])
        
        labelvec = rttm2vadvec(rttm = label, seg_shift_sec = shift_sec, 
                               label2id = label2id, dur = -1)
        sample['label'] = labelvec

        yield sample


def filter_timestamps(data,
           min_num_frames=100,
           max_num_frames=800,
           frame_shift=10,
           data_type='shard/raw/feat'):
    """ Filter the utterance with very short duration and random chunk the
        utterance with very long duration.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            min_num_frames: minimum number of frames of acoustic features
            max_num_frames: maximum number of frames of acoustic features
            frame_shift: the frame shift of the acoustic features (ms)
        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    import copy
    for sample in data:
        assert 'key' in sample
        assert 'spk' in sample
        label = sample['spk']
        new_label = copy.deepcopy(label)

        if data_type == 'feat':
            assert 'feat' in sample
            assert 'sample_rate' in sample
            feat = sample['feat']
            if len(feat) < min_num_frames:
                continue
            elif len(feat) > max_num_frames:
                #TODO
                raise NotImplementedError("Note impelmented chunk for frames yet.")
                feat, new_label = get_random_chunk_timestamps(feat, label, 
                                                              max_num_frames, 
                                                              sample['sample_rate'])
            sample['feat'] = feat
            sample['spk'] = label
        else:
            assert 'sample_rate' in sample
            assert 'wav' in sample
            sample_rate = sample['sample_rate']
            wav = sample['wav'][0]

            min_len = int(frame_shift / 1000 * min_num_frames * sample_rate)
            max_len = int(frame_shift / 1000 * max_num_frames * sample_rate)

            if len(wav) < min_len:
                continue
            elif len(wav) > max_len:
                wav, new_label = get_random_chunk_timestamps(wav, label, max_len, 
                                                             sample['sample_rate'])
            sample['wav'] = wav.unsqueeze(0)
            sample['spk'] = new_label

        yield sample



def random_chunk_timestamps(data, chunk_len, data_type='shard/raw/feat'):
    """ Random chunk the data into chunk_len

        Args:
            data: Iterable[{key, wav/feat, label}]
            chunk_len: chunk length for each sample

        Returns:
            Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        print(sample['key'])
        if(sample['key'] == 'LA_T_5795459'):
            pass
        assert 'key' in sample
        assert 'spk' in sample
        label = sample['spk']

        if data_type == 'feat':
            assert 'feat' in sample
            feat = sample['feat']
            feat, new_label = get_random_chunk_timestamps(feat, label, chunk_len, sample['sample_rate'])
            sample['feat'] = feat
        else:
            assert 'wav' in sample
            wav = sample['wav'][0]
            wav, new_label = get_random_chunk_timestamps(wav, label, chunk_len, sample['sample_rate'])
            sample['wav'] = wav.unsqueeze(0)
            sample['spk'] = new_label
        yield sample

def update_label_with_rttm(data, rttm):
    #rttm = get_rttm(rttm_file)
    for sample in data:
        assert 'key' in sample
        assert 'spk' in sample
        sample['spk'] = rttm[sample['key']]
        yield sample


def clean_batch(data):
    for sample in data:
        if('feat' in sample):
             yield{
                 'key': sample['key'],
                 'feat': sample['feat'],
                 'label': sample['label']
             }

        else:    
             yield{
                 'key': sample['key'],
                 'wav': sample['wav'],
                 'label': sample['label']
             }
