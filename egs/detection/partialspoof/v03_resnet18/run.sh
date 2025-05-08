#!/bin/bash

# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#           2025 Johan Rohdin, Lin Zhang (rohdin@fit.vut.cz, partialspoof@gmail.com)

set -x
. ./path.sh || exit 1

stage=3
stop_stage=3

PS_dir=/export/fs05/lzhan268/workspace/PUBLIC/PartialSpoof/database
data=data/partialspoof # data folder
data_type="shard"  # shard/raw

#config=conf/resnet_noaug_nosample.yaml
#exp_dir=exp/exp/ResNet18-TSTP-emb256-fbank80-wholeutt_nosample-aug0-spFalse-saFalse-Softmax-SGD-epoch100
config=conf/resnet.yaml #wespeaker version 
exp_dir=exp/ResNet18_AugNonoise_F200
gpus="[0]"
num_avg=10 # how many models you want to average
checkpoint=
score_norm_method="asnorm"  # asnorm/snorm
top_n=300

# setup for large margin fine-tuning
lm_config=conf/campplus_lm.yaml

. tools/parse_options.sh || exit 1

# 1. preparing data folder for partialspoof: wav.scp, utt2cls, cls2utt, reco2dur
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh ${PS_dir} ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  # We don't use VAD here but I think the VAD above anyway covers the full utterances
  for dset in train dev eval;do
      if [ $data_type == "shard" ]; then
          python tools/make_shard_list.py --num_utts_per_shard 1000 \
              --num_threads 8 \
              --prefix shards \
              --shuffle \
              ${data}/$dset/wav.scp ${data}/$dset/utt2cls \
              ${data}/$dset/shards ${data}/$dset/shard.list
      else
          python tools/make_raw_list.py --vad_file ${data}/$dset/vad ${data}/$dset/wav.scp \
              ${data}/$dset/utt2cls ${data}/$dset/raw.list
      fi
  done

  #TODO: wespeaker doesn't support multi-channel wavs.
  #MUSAN_dir=/export/fs05/arts/dataset/musan
  #find ${MUSAN_dir} -name "*.wav" | awk -F"/" '{print $NF,$0}' | sort > data/musan/wav.scp
  #RIRs_dir=/export/fs05/arts/dataset/RIRS_NOISES/RIRS_NOISES
  #find ${RIRs_dir} -name "*.wav" | awk -F"/" '{print $NF,$0}' | sort > data/rirs/wav.scp
  # Convert all musan data to LMDB. But note that lmdb does not work on NFS!
  python tools/make_lmdb.py data/musan/wav.scp ${HOME}/local_lmdb/musan/lmdb 
  rsync -av ${HOME}/local_lmdb/musan/lmdb data/musan/lmdb
  # Convert all rirs data to LMDB
  python tools/make_lmdb.py data/rirs/wav.scp ${HOME}/local_lmdb/rirs/lmdb
  rsync -av ${HOME}/local_lmdb/rirs/lmdb data/rirs/lmdb
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=1
  if [[ $(hostname -f) == *fit.vutbr.cz   ]]; then
     gpus=$(python -c "from sys import argv; from safe_gpu import safe_gpu; safe_gpu.claim_gpus(int(argv[1])); print( safe_gpu.gpu_owner.devices_taken )" $num_gpus | sed "s: ::g")
  fi
    ##num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
    #python -m pdb \
    torchrun --rdzv_backend=c10d --rdzv_endpoint=$(hostname):$((RANDOM)) --nnodes=1 --nproc_per_node=$num_gpus \
      wedefense/bin/train.py --config $config \
        --exp_dir ${exp_dir} \
        --gpus $gpus \
        --num_avg ${num_avg} \
        --data_type "${data_type}" \
        --train_data ${data}/train/${data_type}.list \
        --train_label ${data}/train/utt2cls \
        ${checkpoint:+--checkpoint $checkpoint}
        #--reverb_data data/rirs/lmdb \
        #--noise_data data/musan/lmdb \
	#TODO, also move from local/extract_emb.sh
fi

if [ ${stage} -ge 4 ] && [ ${stop_stage} -le 6 ]; then
#  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
#  python wedefense/bin/average_model.py \
#    --dst_model $avg_model \
#    --src_path $exp_dir/models \
#    --num ${num_avg}
#
  model_path=$avg_model
#
#  echo "Extract embeddings ..."
#  num_gpus=1
#  if [[ $(hostname -f) == *fit.vutbr.cz   ]]; then
#     gpus=$(python -c "from sys import argv; from safe_gpu import safe_gpu; safe_gpu.claim_gpus(int(argv[1])); print( safe_gpu.gpu_owner.devices_taken )" $num_gpus | sed "s: ::g")
#  fi
#
#  local/extract_emb.sh \
#     --exp_dir $exp_dir --model_path $model_path \
#     --nj $num_gpus --gpus $gpus --data_type $data_type --data ${data}

  # Stage 5 & 6
  #TODO 1. move out from score_cm.sh 2. check saving folder, clean code.
  echo "Score ..."
  ./local/score_cm.sh \
	  --stage ${stage} \
	  --stop_stage ${stop_stage} \
	  --data ${data}
	  --exp_dir ${exp_dir} \
	  --model_path ${model_path} \
	  --num_classes 2 
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    #conda activate /mnt/matylda6/rohdin/conda/asv_spoof_5_evaluation_package
  for dset in dev eval; do
    # Preparing trails
    # filename        cm-label
    echo "filename cm-label" > ${data}/${dset}/cm_key_file.txt	  
    cat ${data}/${dset}/utt2cls >> ${data}/${dset}/cm_key_file.txt
    sed -i "s/ /\t/g" ${data}/${dset}/cm_key_file.txt

    echo "Measuring " $dset
    python ../../metrics_asvspoof5/evaluation.py  \
	--m t1 \
	--cm ${exp_dir}/embeddings/${dset}/llr.txt \
	--cm_key ${data}/${dset}/cm_key_file.txt
  done
    #conda deactivate
fi

exit 1

