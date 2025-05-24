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

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['src']
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'wget -q -O - {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))


def tar_file_and_group(data):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, wav, spk, sample_rate}]
    """
    for sample in data:
        assert 'stream' in sample
        stream = tarfile.open(fileobj=sample['stream'], mode="r:*")
        prev_prefix = None
        example = {}
        valid = True
        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind('.')
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prefix != prev_prefix:
                example['key'] = prev_prefix
                if valid:
                    yield example
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if postfix in ['spk']:
                        example[postfix] = file_obj.read().decode(
                            'utf8').strip()
                    elif postfix in AUDIO_FORMAT_SETS:
                        waveform, sample_rate = torchaudio.load(file_obj)
                        example['wav'] = waveform
                        example['sample_rate'] = sample_rate
                    else:
                        example[postfix] = file_obj.read()
                except Exception as ex:
                    valid = False
                    logging.warning('error to parse {}'.format(name))
            prev_prefix = prefix
        if prev_prefix is not None:
            example['key'] = prev_prefix
            yield example
        stream.close()
        if 'process' in sample:
            sample['process'].communicate()
        sample['stream'].close()


def parse_raw(data):
    """ Parse key/wav/spk from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/spk

        Returns:
            Iterable[{key, wav, spk, sample_rate}]
    """

    def read_audio(wav):
        if wav.endswith('|'):
            p = Popen(wav[:-1], shell=True, stdout=PIPE)
            data = p.stdout.read()
            waveform, sample_rate = torchaudio.load(io.BytesIO(data))
        else:
            waveform, sample_rate = torchaudio.load(wav)
        return waveform, sample_rate

    def apply_vad(waveform, sample_rate, vad):
        voice_part_list = []
        for start, end in vad:
            start, end = float(start), float(end)
            start, end = int(start * sample_rate), int(end * sample_rate)
            voice_part_list.append(waveform[:, start:end])
        waveform = torch.cat(voice_part_list, dim=1)
        return waveform, sample_rate

    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'wav' in obj
        assert 'spk' in obj
        key = obj['key']
        wav_file = obj['wav']
        spk = obj['spk']
        try:
            waveform, sample_rate = read_audio(wav_file)
            if 'vad' in obj:
                waveform, sample_rate = apply_vad(waveform, sample_rate,
                                                  obj['vad'])
            example = dict(key=key,
                           spk=spk,
                           wav=waveform,
                           sample_rate=sample_rate)
            yield example
        except Exception as ex:
            logging.warning('Failed to read {}'.format(wav_file))


def parse_feat(data):
    """ Parse key/feat/spk from json line

        Args:
            data: Iterable[str], str is a json line has key/feat/spk

        Returns:
            Iterable[{key, feat, spk}]
    """
    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'feat' in obj
        assert 'spk' in obj
        key = obj['key']
        feat_ark = obj['feat']
        spk = obj['spk']
        try:
            feat = torch.from_numpy(kaldiio.load_mat(feat_ark))
            example = dict(key=key, spk=spk, feat=feat)
            yield example
        except Exception as ex:
            logging.warning('Failed to load {}'.format(feat_ark))


def shuffle(data, shuffle_size=2500):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, wav/feat, spk}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, wav/feat, spk}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def spk_to_id(data, spk2id):
    """ Parse spk id

        Args:
            data: Iterable[{key, wav/feat, spk}]
            spk2id: Dict[str, int] e.g.:{'bonafide': 1, 'spoof': 0}

        Returns:
            Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert 'spk' in sample
        if sample['spk'] in spk2id:
            label = spk2id[sample['spk']]
        else:
            label = -1
        sample['label'] = label
        yield sample

def spk_to_id_timestamps(data, label2id):
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

def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.
        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate
        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample


def speed_perturb(data, num_spks):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    speeds = [1.0, 0.9, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed_idx = random.randint(0, 2)
        if speed_idx > 0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speeds[speed_idx])], ['rate',
                                                     str(sample_rate)]])
            sample['wav'] = wav
            sample['label'] = sample['label'] + num_spks * speed_idx

        yield sample


def get_random_chunk(data, chunk_len):
    """ Get random chunk

        Args:
            data: torch.Tensor (random len)
            chunk_len: chunk length

        Returns:
            torch.Tensor (exactly chunk_len)
    """
    data_len = len(data)
    data_shape = data.shape
    # random chunk
    if data_len >= chunk_len:
        chunk_start = random.randint(0, data_len - chunk_len)
        data = data[chunk_start:chunk_start + chunk_len]
        # re-clone the data to avoid memory leakage
        if type(data) == torch.Tensor:
            data = data.clone()
        else:  # np.array
            data = data.copy()
    else:
        # padding
        repeat_factor = chunk_len // data_len + 1
        repeat_shape = repeat_factor if len(data_shape) == 1 else (
            repeat_factor, 1)
        if type(data) == torch.Tensor:
            data = data.repeat(repeat_shape)
        else:  # np.array
            data = np.tile(data, repeat_shape)
        data = data[:chunk_len]

    return data

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


def filter(data,
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
    for sample in data:
        assert 'key' in sample
        assert 'spk' in sample
        label = sample['spk']
        new_label = label.copy()

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


def random_chunk(data, chunk_len, data_type='shard/raw/feat'):
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


def add_reverb_noise(data,
                     reverb_source,
                     noise_source,
                     resample_rate=16000,
                     aug_prob=0.6):
    """ Add reverb & noise aug

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            reverb_source: reverb LMDB data source
            noise_source: noise LMDB data source
            resample_rate: resample rate for reverb/noise data
            aug_prob: aug probability

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'wav' in sample
        assert 'key' in sample
        if aug_prob > random.random():
            aug_type = random.randint(1, 2)
            if aug_type == 1:
                # add reverberation
                audio = sample['wav'].numpy()[0]
                audio_len = audio.shape[0]

                _, rir_data = reverb_source.random_one()
                rir_sr, rir_audio = wavfile.read(io.BytesIO(rir_data))
                rir_audio = rir_audio.astype(np.float32)
                if rir_sr != resample_rate:
                    rir_audio = signal.resample(
                        rir_audio,
                        int(len(rir_audio) / rir_sr * resample_rate))
                rir_audio = rir_audio / np.sqrt(np.sum(rir_audio**2))
                print(sample['key'], audio.shape, rir_audio.shape, audio_len)
                out_audio = signal.convolve(audio, rir_audio,
                                            mode='full')[:audio_len]
            else:
                # add additive noise
                audio = sample['wav'].numpy()[0]
                audio_len = audio.shape[0]
                audio_db = 10 * np.log10(np.mean(audio**2) + 1e-4)

                key, noise_data = noise_source.random_one()
                if key.startswith('noise'):
                    snr_range = [0, 15]
                elif key.startswith('speech'):
                    snr_range = [10, 30]
                elif key.startswith('music'):
                    snr_range = [5, 15]
                else:
                    snr_range = [0, 15]
                noise_sr, noise_audio = wavfile.read(io.BytesIO(noise_data))
                noise_audio = noise_audio.astype(np.float32) / (1 << 15)
                if noise_sr != resample_rate:
                    # Since the noise audio could be very long, it must be
                    # chunked first before resampled (to save time)
                    noise_audio = get_random_chunk(
                        noise_audio, int(audio_len / resample_rate * noise_sr))
                    noise_audio = signal.resample(noise_audio, audio_len)
                else:
                    noise_audio = get_random_chunk(noise_audio, audio_len)
                noise_snr = random.uniform(snr_range[0], snr_range[1])
                noise_db = 10 * np.log10(np.mean(noise_audio**2) + 1e-4)
                noise_audio = np.sqrt(10**(
                    (audio_db - noise_db - noise_snr) / 10)) * noise_audio
                out_audio = audio + noise_audio

            # normalize into [-1, 1]
            out_audio = out_audio / (np.max(np.abs(out_audio)) + 1e-4)
            sample['wav'] = torch.from_numpy(out_audio).unsqueeze(0)

        yield sample


def compute_fbank(data,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=1.0):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15) ## Convert from float [-1.0, 1.0] to int16 [-32767, 32767]
        # Only keep key, feat, label
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          sample_frequency=sample_rate,
                          window_type='hamming',
                          use_energy=False)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def apply_cmvn(data, norm_mean=True, norm_var=False):
    """ Apply CMVN

        Args:
            data: Iterable[{key, feat, label}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'key' in sample
        assert 'feat' in sample
        assert 'label' in sample
        mat = sample['feat']
        if norm_mean:
            mat = mat - torch.mean(mat, dim=0)
        if norm_var:
            mat = mat / torch.sqrt(torch.var(mat, dim=0) + 1e-8)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def spec_aug(data, num_t_mask=1, num_f_mask=1, max_t=10, max_f=8, prob=0.6):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            prob: prob of spec_aug

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        if random.random() < prob:
            assert 'feat' in sample
            x = sample['feat']
            assert isinstance(x, torch.Tensor)
            # y = x.clone().detach()
            y = x.detach()  # inplace operation
            max_frames = y.size(0)
            max_freq = y.size(1)
            # time mask
            for i in range(num_t_mask):
                start = random.randint(0, max_frames - 1)
                length = random.randint(1, max_t)
                end = min(max_frames, start + length)
                y[start:end, :] = 0
            # freq mask
            for i in range(num_f_mask):
                start = random.randint(0, max_freq - 1)
                length = random.randint(1, max_f)
                end = min(max_freq, start + length)
                y[:, start:end] = 0
            sample['feat'] = y
        yield sample

def rawboost(data, 
             argo = 5):
    """ Process Rawboost

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: resample rate for data
            argo: argo. tag for rawboost, default value is 5.

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        sample_rate = sample['sample_rate']
        audio = sample['wav'].numpy()[0]
        audio_len = audio.shape[0]

        sample['wav'] = rawboost_util.process_Rawboost_feature(audio, sample_rate, algo)
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
