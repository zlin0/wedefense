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

config=conf/resnet.yaml
exp_dir=exp/ResNet18
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
 # for dset in train dev eval;do
 #     if [ $data_type == "shard" ]; then
 #         python tools/make_shard_list.py --num_utts_per_shard 1000 \
 #             --num_threads 8 \
 #             --prefix shards \
 #             --shuffle \
 #             ${data}/$dset/wav.scp ${data}/$dset/utt2cls \
 #             ${data}/$dset/shards ${data}/$dset/shard.list
 #     else
 #         python tools/make_raw_list.py --vad_file ${data}/$dset/vad ${data}/$dset/wav.scp \
 #             ${data}/$dset/utt2cls ${data}/$dset/raw.list
 #     fi
 # done

  #MUSAN_dir=/export/fs05/arts/dataset/musan
  #find ${MUSAN_dir} -name "*.wav" | awk -F"/" '{print $NF,$0}' | sort > data/musan/wav.scp
  #RIRs_dir=/export/fs05/arts/dataset/RIRS_NOISES/RIRS_NOISES
  #find ${RIRs_dir} -name "*.wav" | awk -F"/" '{print $NF,$0}' | sort > data/rirs/wav.scp
  # Convert all musan data to LMDB
  python tools/make_lmdb.py data/musan/wav.scp data/musan/lmdb
  # Convert all rirs data to LMDB
  python tools/make_lmdb.py data/rirs/wav.scp data/rirs/lmdb
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
    -m pdb \
      wedefense/bin/train.py --config $config \
        --exp_dir ${exp_dir} \
        --gpus $gpus \
        --num_avg ${num_avg} \
        --data_type "${data_type}" \
        --train_data ${data}/train/${data_type}.list \
        --train_label ${data}/train/utt2cls \
        --reverb_data ${data}/rirs/lmdb \
        --noise_data ${data}/musan/lmdb \
        ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
  #python wespeaker/bin/average_model.py \
  #  --dst_model $avg_model \
  #  --src_path $exp_dir/models \
  #  --num ${num_avg}

  model_path=$avg_model
  #if [[ $config == *repvgg*.yaml ]]; then
  #  echo "convert repvgg model ..."
  #  python wespeaker/models/convert_repvgg.py \
  #    --config $exp_dir/config.yaml \
  #    --load $avg_model \
  #    --save $exp_dir/models/convert_model.pt
  #  model_path=$exp_dir/models/convert_model.pt
  #fi

  echo "Extract embeddings ..."
  #num_gpus=2
  #gpus=$(python -c "from sys import argv; from safe_gpu import safe_gpu; safe_gpu.claim_gpus(int(argv[1])); print( safe_gpu.gpu_owner.devices_taken )" $num_gpus | sed "s: ::g")
  #local/extract_vox.sh \
  #  --exp_dir $exp_dir --model_path $model_path \
  #  --nj $num_gpus --gpus $gpus --data_type $data_type --data ${data}

  echo "Extract embeddings ..."
  num_gpus=2
  gpus=$(python -c "from sys import argv; from safe_gpu import safe_gpu; safe_gpu.claim_gpus(int(argv[1])); print( safe_gpu.gpu_owner.devices_taken )" $num_gpus | sed "s: ::g")
  #local/extract_vox_eval.sh \
  #  --exp_dir $exp_dir --model_path $model_path \
  #  --nj $num_gpus --gpus $gpus --data_type $data_type --data ${data}

  #local/extract_vox.sh \
  #  --exp_dir $exp_dir --model_path $model_path \
  #  --nj $num_gpus --gpus $gpus --data_type $data_type --data ${data}

  local/extract_vox_eval_extra.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj $num_gpus --gpus $gpus --data_type $data_type --data ${data}


fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Score ..."
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Score norm ..."
  local/score_norm.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method $score_norm_method \
    --cohort_set vox2_dev \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Score calibration ..."
  local/score_calibration.sh \
    --stage 1 --stop-stage 5 \
    --score_norm_method $score_norm_method \
    --calibration_trial "vox2_cali.kaldi" \
    --cohort_set vox2_dev \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Export the best model ..."
  python wespeaker/bin/export_jit.py \
    --config $exp_dir/config.yaml \
    --checkpoint $exp_dir/models/avg_model.pt \
    --output_file $exp_dir/models/final.zip
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  echo "Large margin fine-tuning ..."
  lm_exp_dir=${exp_dir}-LM
  mkdir -p ${lm_exp_dir}/models
  # Use the pre-trained average model to initialize the LM training
  cp ${exp_dir}/models/avg_model.pt ${lm_exp_dir}/models/model_0.pt
  bash run.sh --stage 3 --stop_stage 8 \
      --data ${data} \
      --data_type ${data_type} \
      --config ${lm_config} \
      --exp_dir ${lm_exp_dir} \
      --gpus $gpus \
      --num_avg 1 \
      --checkpoint ${lm_exp_dir}/models/model_0.pt \
      --trials "$trials" \
      --score_norm_method ${score_norm_method} \
      --top_n ${top_n}
fi
