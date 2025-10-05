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

Prepare data folder for database: wav.scp, utt2lab, lab2utt, reco2dur, rttm.

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
    if real_or_fake == 'spoof' and av_info['fake_periods']:
        # fake periods exist
        for fake_start, fake_end in av_info['fake_periods']:
            # fake
            if (start_time < fake_start):
                one_segment = f"""SPEAKER {av_info['file']} 1 {start_time:7.3f} {(fake_start - start_time):7.3f} <NA> <NA> bonafide <NA> <NA>"""
                one_rttm.append(one_segment)
            one_segment = f"""SPEAKER {av_info['file']} 1 {fake_start:7.3f} {(fake_end - fake_start):7.3f} <NA> <NA> spoof <NA> <NA>"""
            one_rttm.append(one_segment)
            start_time = fake_end + 1e-6  # avoid overlap with fake segment
        if start_time < av_info['duration']:
            # real
            one_segment = f"""SPEAKER {av_info['file']} 1 {start_time:7.3f} {(av_info['duration'] - start_time):7.3f} <NA> <NA> bonafide <NA> <NA>"""
            one_rttm.append(one_segment)
    elif real_or_fake == 'bonafide' or (av_info['fake_periods'][0] == 0
                                        and av_info['fake_periods'][1]
                                        == av_info['duration']):
        # real
        one_segment = f"""SPEAKER {av_info['file']} 1 {start_time:7.3f} {(av_info['duration'] - start_time):7.3f} <NA> <NA> bonafide <NA> <NA>"""
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
        dset: {
            'wav_scp': [],
            'utt2lab': [],
            'reco2dur': [],
            'rttm': []
        }
        for dset in ["train", "dev", "test"]
    }

    for dset in ["train", "dev", "test"]:
        os.makedirs(f"{save_dir}/{dset}", exist_ok=True)

    print("Processing JSON data...")
    for av_info in tqdm(data):
        dset = av_info['split']
        if dset not in outputs:
            continue

        real_or_fake = 'spoof' if av_info['n_fakes'] > 0 else 'bonafide'

        wav_path = os.path.join(wav_dir, av_info['file'])

        outputs[dset]['wav_scp'].append(f"{av_info['file']} {wav_path}")
        outputs[dset]['utt2lab'].append(f"{av_info['file']} {real_or_fake}")
        outputs[dset]['reco2dur'].append(
            f"{av_info['file']} {av_info['duration']}")
        outputs[dset]['rttm'].extend(get_utt_rttm(av_info, real_or_fake))

    print("Writing Kaldi-style files...")
    for dset, files in outputs.items():
        for file_type, lines in files.items():
            if (file_type == 'wav_scp'):
                with open(f"{save_dir}/{dset}/wav.scp", "w") as f:
                    f.write("\n".join(lines) + "\n")
            else:
                with open(f"{save_dir}/{dset}/{file_type}", "w") as f:
                    f.write("\n".join(lines) + "\n")


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

    read_json_file(args.json_file, args.save_dir)


def debug():
    json_file_path = '/export/fs05/arts/dataset/LAV-DF/LAV-DF/metadata.json'
    wav_dir = "/export/fs05/arts/dataset/LAV-DF/LAV-DF/"
    save_dir = './data/'
    read_json_file(json_file_path, save_dir, wav_dir)


if __name__ == "__main__":
    main()
    # debug()
