#!/bin/bash
set -ex


#1. json_to_rttm_kaldi.py
wav_dir=/export/fs06/arts/dataset/AV-Deepfake1M
data=data/avdeepfake1m # data folder
data_type="raw"  # shard/raw
#DSETs="train val"
DSETs="val"

  for dset in ${DSETs} ;do
      mkdir -p ./${data}
      python ./local/json_to_rttm_kaldi.py \
      --json_file "${wav_dir}/${dset}_metadata.json" \
      --save_dir "${data}/${dset}" \
      --wav_dir "${wav_dir}/${dset}"
        # create lab2utt
      ./tools/utt2lab_to_lab2utt.pl ${data}/${dset}/utt2lab >${data}/${dset}/lab2utt
  done


# mv data/train data/train_all
# mv data/val data/val_all

#2. extract those we need only:
# fake_video_fake_audio.mp4 (has the same audio as real_video_fake_audio.mp4 )
# real.mp4 (has the same audio as fake_video_real_audio.mp4)

for dset in ${DSETs}; do
    rm -rf ${data}/${dset}_audioonly
    mkdir -p ${data}/${dset}_audioonly
    # for file in metadata.scp; do #TODO: remove raw.list
    # for file in raw.list  rttm  utt2dur  utt2lab  wav.scp; do
    for file in metadata.scp  rttm  utt2dur  utt2lab  wav.scp; do
        for x in fake_video_fake_audio fake_video_real_audio; do
            grep "$x" ${data}/${dset}/${file} >> ${data}/${dset}_audioonly/${file}
        done
    done
    ./tools/utt2lab_to_lab2utt.pl ${data}/${dset}_audioonly/utt2lab >${data}/${dset}_audioonly/lab2utt
done
