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

import argparse

from wedefense.cli.model import load_model


def get_args():
    parser = argparse.ArgumentParser(description='WeDefense CLI')
    parser.add_argument('-t', '--task', choices=['detection', 'localization'], default='detection', help='Task type')
    parser.add_argument('-m', '--model_dir', default=None, help='Specify your own model directory')
    parser.add_argument('-i', '--model_id', choices=["MHFA_wav2vec2_large"], default='MHFA_wav2vec2_large', help='')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for computation (e.g., cpu, cuda:0)')
    parser.add_argument('--resample_rate', type=int, default=16000, help='Resample rate for audio processing')
    parser.add_argument('--audio_file', type=str, help='Path to the audio file')
    args = parser.parse_args()
    return args



def main():
    args = get_args()
    print(args)
    model = load_model(args.model_id, args.model_dir)

    model.set_device(args.device)
    model.set_resample_rate(args.resample_rate)

    if args.task == 'detection':
        model.detection(args.audio_file)
    elif args.task == 'localization':
        model.localization()


if __name__ == "__main__":
    main()
