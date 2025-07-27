# Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
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
import math
import pickle

import os
import lmdb
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('in_scp_file', help='input scp file')
    parser.add_argument('out_lmdb', help='output lmdb')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    try:
        # Attempt to open LMDB environment
        print("Opening LMDB at:", args.out_lmdb)
        os.makedirs(os.path.dirname(args.out_lmdb), exist_ok=True)
        db = lmdb.open(args.out_lmdb, map_size=int(math.pow(1024, 4)))  # 1TB,
        print("LMDB opened successfully.")
    except Exception as e:
        # Use a local path such as /tmp or $HOME for LMDB storage.
        print("Failed to open LMDB.")
        print(
            "If the program hangs or freezes here, please check if the path is located on an NFS-mounted filesystem."
        )
        print(
            "LMDB may not work properly on NFS due to mmap and file lock issues."
        )
        print(f"Error message: {e}")
    # txn is for Transaciton
    txn = db.begin(write=True)
    keys = []
    with open(args.in_scp_file, 'r', encoding='utf8') as fin:
        lines = fin.readlines()
        for i, line in enumerate(tqdm(lines)):
            arr = line.strip().split()
            assert len(arr) == 2
            key, wav = arr[0], arr[1]
            keys.append(key)
            with open(wav, 'rb') as fin:
                data = fin.read()
            txn.put(key.encode(), data)
            # Write flush to disk
            if i % 100 == 0:
                txn.commit()
                txn = db.begin(write=True)
    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
    db.sync()
    db.close()


if __name__ == '__main__':
    main()
