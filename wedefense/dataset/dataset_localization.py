# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2022 Hongji Wang (jijijiang77@gmail.com)
#               2023 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2025 Lin Zhang, Xin Wang (lzhang.as@gmail.com, wangxin@nii.ac.jp)
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

import random

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from wedefense.utils.file_utils import read_lists, read_table, read_json_list
from wedefense.dataset.lmdb_data import LmdbData
import wedefense.dataset.processor as processor
import wedefense.dataset.shuffle_random_tools as shuffle_tools

class Processor(IterableDataset):
    # https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset

    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:
    """ Sampler for distributed training
    """
    def __init__(self, shuffle=True, partition=True, block_shuffle_size=0):
        """ Initialzie DistributedSampler

            Args:
                shuffle (bool): whether shuffle data list. 
                    Default to True
                partition (bool): whether divide file list based on world_size. 
                    Default to True
                block_shuffle_size (int): if larger than 0, shuffle by block
                    [1,2,3,4,5,6] -> shuffle_blck_size with 3 -> [3,1,2,5,4,6]
                    Default to 0
        """
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition
        self.block_shuffle_size = block_shuffle_size

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        if self.partition:
            if self.shuffle and self.block_shuffle_size > 0:
                # shuffle within blocks
                # e.g., [1,2,3,4,5,6], block_size=3 -> [3,1,2,5,4,6]
                shuffle_tools.f_shuffle_in_block_inplace(data, self.block_shuffle_size, self.epoch)
                # shuffle blocks
                # e.g., [3,1,2,5,4,6], block_size=3 -> [5,4,6,3,1,2]
                shuffle_tools.f_shuffle_blocks_inplace(data, self.block_shuffle_size, self.epoch)
                
            elif self.shuffle:
                # default shuffle
                random.Random(self.epoch).shuffle(data)
            else:
                pass
            
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data


class DataList(IterableDataset):
    """
    Dataset based on torch IterableDataset
    """
    
    def __init__(self,
                 lists,
                 shuffle=True,
                 partition=True,
                 repeat_dataset=True,
                 block_shuffle_size = 0):
        """ Initialize DataList.

            Args:
                lists (list): list of data files
                shuffle (bool): whether shuffle data list. 
                    Default to True
                partition (bool): whether divide file list based on world_size. 
                    Default to True
                repeat_dataset (bool): whether repeat_dataset during loading.
                    Default to True
                block_shuffle_size (int): if larger than 0, shuffle by block
                    [1,2,3,4,5,6] -> shuffle_blck_size with 3 -> [3,1,2,5,4,6]
                    Default to 0

            When block_shuffle_size > 0, we assume that lists is a sorted list of 
            file paths, wherein the sorting is based on the duration of each file.
        """
        self.lists = lists
        self.repeat_dataset = repeat_dataset
        self.sampler = DistributedSampler(shuffle, partition, block_shuffle_size)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        if not self.repeat_dataset:
            for index in indexes:
                data = dict(src=self.lists[index])
                data.update(sampler_info)
                yield data
        else:
            indexes_len = len(indexes)
            counter = 0
            while True:
                index = indexes[counter % indexes_len]
                counter += 1
                data = dict(src=self.lists[index])
                data.update(sampler_info)
                yield data


def Dataset(data_type,
            data_list_file,
            configs,
            spk2id_dict,
            whole_utt=False,
            reverb_lmdb_file=None,
            noise_lmdb_file=None,
            repeat_dataset=True,
            data_dur_file=None,
            reco2timestamps_dict=None,
            block_shuffle_size = 0,
            output_reso = 0.01):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw/feat file level. The second is local shuffle
        at training samples level.

        Args:
            data_type(str): shard/raw/feat
            data_list_file: data list file
            configs: dataset configs
            spk2id_dict: spk2id dict
            reverb_lmdb_file: reverb data source lmdb file
            noise_lmdb_file: noise data source lmdb file
            whole_utt: use whole utt or random chunk
            repeat_dataset: True for training while False for testing
            data_dur_file (str): path to the utterance duration file
            reco2timestamps_dict (dict): dict to save map between recoid to rttm
            block_shuffle_size (int): size of block shuffle. 
                    Default to 0, no block shuffle
    """
    assert data_type in ['shard', 'raw', 'feat']
    frontend_type = configs.get('frontend', 'fbank')
    frontend_args = frontend_type + "_args"

    # lists of file
    lists = read_lists(data_list_file)
        
    # Both block_fhuffle and localization require duration information.
    assert data_dur_file is not None, "utt2dur is required"
    # load duration
    utt2dur = {x[0]:float(x[1]) for x in read_table(data_dur_file)}
    # block_shuffle is to be used, sort the file list based on duration
    if block_shuffle_size > 0:
        assert data_type == 'raw', "block shuffle requires raw data type"

        # data_list_file is not necessarily aligned with the utt2dur file
        js = read_json_list(data_list_file)
        u2d = [utt2dur[x['key']] if x['key'] in uttdur else -1 for x in js]

        assert len(u2d) == len(lists), \
            "#.lines unequal {:s} {:s}".format(data_dur_file, data_list_file)
        # re-order lists based on utterance duration
        lists = [x[0] for x in sorted(zip(lists, u2d), key=lambda x: x[1])]

    shuffle = configs.get('shuffle', False)
    #   set in train.py to keep consistent with wespeaker.
    # whole_utt = configs.get('whole_utt', False)
    # Global shuffle
    dataset = DataList(
        lists,
        shuffle=shuffle,
        repeat_dataset=repeat_dataset,
        block_shuffle_size=block_shuffle_size)
    
    if data_type == 'shard':
        dataset = Processor(dataset, processor.url_opener)
        dataset = Processor(dataset, processor.tar_file_and_group)
    elif data_type == 'raw':
        dataset = Processor(dataset, processor.parse_raw)
    else:
        dataset = Processor(dataset, processor.parse_feat)

    dataset = Processor(dataset, processor.update_label_with_rttm, reco2timestamps_dict)

    if configs.get('filter', True):
        # Filter the data with unwanted length
        filter_conf = configs.get('filter_args', {})
        dataset = Processor(dataset,
                            processor.filter_timestamps,
                            frame_shift=configs[frontend_args].get(
                                'frame_shift', 10),
                            data_type=data_type,
                            **filter_conf)

    # Local shuffle
    #  if block_shuffle_size is on, shuffling is done using Global shuffle
    if shuffle and block_shuffle_size == 0:
        dataset = Processor(dataset, processor.shuffle,
                            **configs['shuffle_args'])


    if data_type == 'feat':
        if not whole_utt:
            # random chunk
            chunk_len = num_frms = configs.get('num_frms', 200)
            dataset = Processor(dataset, processor.random_chunk, chunk_len,
                                'feat')
    else:
        # resample
        resample_rate = configs.get('resample_rate', 16000)
        dataset = Processor(dataset, processor.resample, resample_rate)
        # speed perturb
        speed_perturb_flag = configs.get('speed_perturb', True)
        if speed_perturb_flag:
            dataset = Processor(dataset, processor.speed_perturb,
                                len(spk2id_dict))
        if not whole_utt:
            # random chunk
            num_frms = configs.get('num_frms', 200)
            frame_shift = configs[frontend_args].get('frame_shift', 10)
            frame_length = configs[frontend_args].get('frame_length', 25)
            chunk_len = ((num_frms - 1) * frame_shift +
                         frame_length) * resample_rate // 1000
            dataset = Processor(dataset, processor.random_chunk, chunk_len,
                                data_type)
            
        # process Rawboost:
        rawboost_flag = configs.get('rawboost', False)
        if (rawboost_flag):
            dataset = Processor(dataset, processor.rawboost, algo = 5)
        # add reverb & noise
        aug_prob = configs.get('aug_prob', 0.6)
        if (reverb_lmdb_file and noise_lmdb_file) and (aug_prob > 0.0):
            reverb_data = LmdbData(reverb_lmdb_file)
            noise_data = LmdbData(noise_lmdb_file)
            dataset = Processor(dataset, processor.add_reverb_noise,
                                reverb_data, noise_data, resample_rate,
                                aug_prob)
        # compute fbank
        if frontend_type == 'fbank':
            dataset = Processor(dataset, processor.compute_fbank,
                                **configs['fbank_args'])


    # spk2id
    dataset = Processor(dataset, processor.spk_to_id_timestamps, spk2id_dict)
    # Convert timestamps to frame-level label vector
    # Put to the final step after all chunk/shuffle.
    dataset = Processor(dataset, processor.timestamps_to_labelvec, 
                        output_reso, spk2id_dict, utt2dur) #we use 0.01sec as unit.

    # keep timestamps will get error when collect.py:138 
    dataset = Processor(dataset, processor.clean_batch) 

    # !!!IMPORTANT NOTICE!!!
    # To support different frontends (including ssl pretrained models),
    # we have to move apply_cmvn and spec_aug out of the dataset pipeline
    # which runs totally in cpus.
    # These two modules are now used in wedefense/utils/executor.py (train)
    # and wedefense/bin/extract.py (test), which runs in gpus.
    '''
    # apply cmvn
    dataset = Processor(dataset, processor.apply_cmvn)

    # spec augmentation
    spec_aug_flag = configs.get('spec_aug', True)
    if spec_aug_flag:
        dataset = Processor(dataset, processor.spec_aug,
                            **configs['spec_aug_args'])
    '''
    return dataset
