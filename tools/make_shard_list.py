# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Shanghai Jiaotong University (authors: Zhengyang Chen)
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
import io
import logging
import os
import random
import tarfile
import time
import multiprocessing
import subprocess
from scipy.io import wavfile
import numpy as np
import struct

AUDIO_FORMAT_SETS = set(
    ['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma', 'mp4'])


def write_wav_to_bytesio(audio_data, sample_rate):
    audio_data = audio_data.astype(np.int16)

    with io.BytesIO() as wav_stream:
        # WAV header values
        num_channels = 1  # Mono audio
        bytes_per_sample = 2  # Assuming 16-bit audio

        # Write WAV header
        wav_stream.write(b'RIFF')
        wav_stream.write(b'\x00\x00\x00\x00')  # Placeholder for file size
        wav_stream.write(b'WAVE')

        # Write format chunk
        wav_stream.write(b'fmt ')
        wav_stream.write(struct.pack('<I', 16))  # Chunk size
        wav_stream.write(struct.pack('<H', 1))  # Audio format (PCM)
        wav_stream.write(struct.pack('<H', num_channels))
        wav_stream.write(struct.pack('<I', sample_rate))
        wav_stream.write(
            struct.pack('<I', sample_rate * num_channels * bytes_per_sample))
        wav_stream.write(struct.pack('<H', num_channels * bytes_per_sample))
        wav_stream.write(struct.pack('<H', bytes_per_sample * 8))

        # Write data chunk header
        wav_stream.write(b'data')
        wav_stream.write(struct.pack('<I', len(audio_data) * bytes_per_sample))

        # Write audio data
        for sample in audio_data:
            wav_stream.write(struct.pack('<h', sample))

        # Go back and fill in the correct file size
        file_size = wav_stream.tell()
        wav_stream.seek(4)
        wav_stream.write(struct.pack('<I', file_size - 8))

        # Return the BytesIO stream
        return wav_stream.getvalue()


def apply_vad(wav_data, vad):
    sr, audio = wavfile.read(io.BytesIO(wav_data))

    voice_part_list = []
    for start, end in vad:
        start, end = float(start), float(end)
        start, end = int(start * sr), int(end * sr)
        voice_part_list.append(audio[start:end])
    audio = np.concatenate(voice_part_list)
    voiced_wav_data = write_wav_to_bytesio(audio, sr)
    return voiced_wav_data


def write_tar_file(data_list, tar_file, index=0, total=1):
    logging.info('Processing {} {}/{}'.format(tar_file, index, total))
    read_time = 0.0
    write_time = 0.0
    with tarfile.open(tar_file, "w") as tar:
        for item in data_list:
            if len(item) == 3:
                key, lab, wav = item
                vad = None
            else:
                key, lab, wav, vad = item

            if wav.endswith('|'):
                suffix = 'wav'
            else:
                suffix = wav.split('.')[-1]
            assert suffix in AUDIO_FORMAT_SETS

            ts = time.time()
            if wav.endswith('|'):
                p = subprocess.Popen(wav[:-1],
                                     shell=True,
                                     stdout=subprocess.PIPE)
                data = p.stdout.read()
            else:
                with open(wav, 'rb') as fin:
                    data = fin.read()

            if vad is not None:
                data = apply_vad(data, vad)
            read_time += (time.time() - ts)
            assert isinstance(lab, str)
            ts = time.time()
            lab_file = key + '.lab'
            lab = lab.encode('utf8')
            lab_data = io.BytesIO(lab)
            lab_info = tarfile.TarInfo(lab_file)
            lab_info.size = len(lab)
            tar.addfile(lab_info, lab_data)

            wav_file = key + '.' + suffix
            wav_data = io.BytesIO(data)
            wav_info = tarfile.TarInfo(wav_file)
            wav_info.size = len(data)
            tar.addfile(wav_info, wav_data)
            write_time += (time.time() - ts)
        logging.info('read {} write {}'.format(read_time, write_time))


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_utts_per_shard',
                        type=int,
                        default=1000,
                        help='num utts per shard')
    parser.add_argument('--num_threads',
                        type=int,
                        default=1,
                        help='num threads for make shards')
    parser.add_argument('--prefix',
                        default='shards',
                        help='prefix of shards tar file')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--shuffle',
                        action='store_true',
                        help='whether to shuffle data')
    parser.add_argument('--vad_file',
                        type=str,
                        help='vad file',
                        default='non_exist')
    parser.add_argument('wav_file', help='wav file')
    parser.add_argument('utt2lab_file', help='utt2lab file')
    parser.add_argument('shards_dir', help='output shards dir')
    parser.add_argument('shards_list', help='output shards list file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    random.seed(args.seed)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    wav_table = {}
    with open(args.wav_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            key = arr[0]  # key = os.path.splitext(arr[0])[0]
            wav_table[key] = ' '.join(arr[1:])

    if os.path.exists(args.vad_file):
        vad_dict = {}
        with open(args.vad_file, 'r', encoding='utf8') as fin:
            for line in fin:
                arr = line.strip().split()
                utt, start, end = arr[-3], arr[-2], arr[-1]
                if utt not in vad_dict:
                    vad_dict[utt] = []
                vad_dict[utt].append((start, end))
    else:
        vad_dict = None

    data = []
    with open(args.utt2lab_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split(maxsplit=1)
            key = arr[0]  # key = os.path.splitext(arr[0])[0]
            lab = arr[1]
            assert key in wav_table
            wav = wav_table[key]
            if vad_dict is None:
                data.append((key, lab, wav))
            else:
                """
                if key not in vad_dict:
                    continue
                vad = vad_dict[key]
                data.append((key, lab, wav, vad))
                """
                if key not in vad_dict:
                    data.append((key, lab, wav))
                else:
                    vad = vad_dict[key]
                    data.append((key, lab, wav, vad))

    if args.shuffle:
        random.shuffle(data)

    num = args.num_utts_per_shard
    chunks = [data[i:i + num] for i in range(0, len(data), num)]
    os.makedirs(args.shards_dir, exist_ok=True)

    # Using thread pool to speedup
    pool = multiprocessing.Pool(processes=args.num_threads)
    shards_list = []
    tasks_list = []
    num_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        tar_file = os.path.join(args.shards_dir,
                                '{}_{:09d}.tar'.format(args.prefix, i))
        shards_list.append(tar_file)
        pool.apply_async(write_tar_file, (chunk, tar_file, i, num_chunks))

    pool.close()
    pool.join()

    with open(args.shards_list, 'w', encoding='utf8') as fout:
        for name in shards_list:
            fout.write(name + '\n')


def debug_write_tar():
    """Debug the write_tar_file function directly"""

    # Create some test data
    test_data = [
        ("test1", "bonafide",
         "/export/fs05/arts/dataset/LAV-DF/LAV-DF/train/000472.mp4"),
        ("test2", "spoof",
         "/export/fs05/arts/dataset/LAV-DF/LAV-DF/train/000473.mp4"),
    ]

    # Test file path
    test_tar_file = "debug_test.tar"

    print("=== Debug write_tar_file function ===")
    print(f"Test data: {test_data}")
    print(f"Output tar file: {test_tar_file}")
    print("=" * 50)

    # Call the function directly - you can set breakpoints here!
    try:
        write_tar_file(test_data, test_tar_file, index=0, total=1)
        print("Function completed successfully!")
    except Exception as e:
        print(f"Function failed: {e}")
        import traceback
        traceback.print_exc()
    args = get_args()


if __name__ == '__main__':
    # debug_write_tar()
    main()
