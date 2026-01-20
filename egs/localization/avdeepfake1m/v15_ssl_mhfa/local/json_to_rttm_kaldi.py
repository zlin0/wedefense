#!/usr/bin/env python

# Copyright (c) 2025 Lin Zhang (partialspoof@gmail.com)
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
#
"""
local/json_to_rttm_kaldi.py

Prepare data folder for database: wav.scp, utt2lab, lab2utt, utt2dur, rttm.

"""

import json
from tqdm import tqdm
import argparse
import os


def get_utt_rttm(av_info, real_or_fake) -> str:
    """
    Convert meta data from json file to rttm file.
    row example for rttm:
      SPEAKER LA_T_1000406 1   0.000   1.100 <NA> <NA> bonafide <NA> <NA>
    """
    start_time = 0
    one_rttm = []
    duration = av_info['audio_frames'] / 16000.0
    if real_or_fake == 'spoof' and av_info['audio_fake_segments']:
        # fake periods exist
        for fake_start, fake_end in av_info['audio_fake_segments']:
            # fake
            if (start_time < fake_start):
                one_segment = f"""SPEAKER {av_info['file']} 1 {start_time:7.3f} {(fake_start - start_time):7.3f} <NA> <NA> bonafide <NA> <NA>"""
                one_rttm.append(one_segment)
            one_segment = f"""SPEAKER {av_info['file']} 1 {fake_start:7.3f} {(fake_end - fake_start):7.3f} <NA> <NA> spoof <NA> <NA>"""
            one_rttm.append(one_segment)
            start_time = fake_end + 1e-6  # avoid overlap with fake segment
        if start_time < duration:
            # real
            one_segment = f"""SPEAKER {av_info['file']} 1 {start_time:7.3f} {(duration - start_time):7.3f} <NA> <NA> bonafide <NA> <NA>"""
            one_rttm.append(one_segment)
    elif real_or_fake == 'bonafide':
        # real
        one_segment = f"""SPEAKER {av_info['file']} 1 {start_time:7.3f} {(duration - start_time):7.3f} <NA> <NA> bonafide <NA> <NA>"""
        one_rttm.append(one_segment)
    else:
        raise ValueError(f"Invalid real_or_fake: {real_or_fake}")

    return one_rttm


def read_json_file(json_file_path, save_dir, wav_dir) -> None:
    """
    {
    "file": "test/000001.mp4",
    "n_fakes": 0,
    "fake_periods": [],
    "timestamps": [
      [
        "not",
        0.0,
        0.2
      ],
      ...
      [
        "life",
        3.8,
        4.0
      ]
    ],
    "duration": 4.224,
    "transcript": "not the point the point is ...",
    "original": null,
    "modify_video": false,
    "modify_audio": false,
    "split": "test",
    "video_frames": 103,
    "audio_channels": 1,
    "audio_frames": 65536
  },
    """
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    outputs = {
        'wav_scp': [],
        'utt2lab': [],
        'utt2dur': [],
        'rttm': [],
        'metadata_scp': []
    }

    os.makedirs(f"{save_dir}", exist_ok=True)

    silent_files = []  # Track files without audio streams

    print("Processing JSON data...")
    for av_info in tqdm(data):

        real_or_fake = 'spoof' if len(
            av_info['audio_fake_segments']) > 0 else 'bonafide'

        wav_path = os.path.join(wav_dir, av_info['file'])

        # Use ffmpeg pipe for mp4 files to ensure consistency
        if wav_path.endswith('.mp4'):
            # Check if file has audio stream using ffprobe
            metadata_path = wav_path.replace('/train/',
                                             '/train_metadata/').replace(
                                                 '.mp4', '.json')
            metadata_path = wav_path.replace('/val/',
                                             '/val_metadata/').replace(
                                                 '.mp4', '.json')

            # Use ffmpeg pipe command for consistent decoding
            # -loglevel quiet: suppress ffmpeg output (only show errors)
            # -map 0:a?: try to map audio stream
            #     (optional, fails gracefully if no audio)
            # wav_path_quoted = shlex.quote(wav_path)
            # wav_path = (
            #     f"ffmpeg -loglevel error -i {wav_path_quoted} "
            #     f"-map 0:a:0? -f wav -acodec pcm_s16le "
            #     f"-ar 16000 -ac 1 - |"
            # )

        outputs['wav_scp'].append(f"{av_info['file']} {wav_path}")
        outputs['metadata_scp'].append(f"{av_info['file']} {metadata_path}")
        outputs['utt2lab'].append(f"{av_info['file']} {real_or_fake}")
        # meta data doesn't conrain duration info.
        outputs['utt2dur'].append(
            f"{av_info['file']} {av_info['audio_frames']/16000.0}")
        outputs['rttm'].extend(get_utt_rttm(av_info, real_or_fake))

    print("Writing Kaldi-style files...")
    for file_type, lines in outputs.items():
        if (file_type == 'wav_scp'):
            with open(f"{save_dir}/wav.scp", "w") as f:
                f.write("\n".join(lines) + "\n")
        elif (file_type == 'metadata_scp'):
            with open(f"{save_dir}/metadata.scp", "w") as f:
                f.write("\n".join(lines) + "\n")
        else:
            with open(f"{save_dir}/{file_type}", "w") as f:
                f.write("\n".join(lines) + "\n")
    print("Finish writing...")


def main():
    parser = argparse.ArgumentParser(description="Process dataset into CSV")
    parser.add_argument(
        '--json_file',
        type=str,
        default='/export/fs05/arts/dataset/LAV-DF/LAV-DF/metadata.json',
        help="Path to the dataset directory")
    parser.add_argument('--save_dir',
                        type=str,
                        default='./data/',
                        help="Directory to save the output CSV files")
    parser.add_argument(
        '--wav_dir',
        type=str,
        default='/export/fs05/arts/dataset/LAV-DF/LAV-DF/',
        help="Parent directory to wav when json only save filename.")
    args = parser.parse_args()

    read_json_file(args.json_file, args.save_dir, args.wav_dir)


def debug():
    json_file_path = '/export/fs05/arts/dataset/AV-Deepfake1M-PlusPlus/train_metadata.json'
    wav_dir = "/export/fs05/arts/dataset/AV-Deepfake1M-PlusPlus/train"
    save_dir = './data/debug'
    read_json_file(json_file_path, save_dir, wav_dir)


if __name__ == "__main__":
    main()
    # debug()
