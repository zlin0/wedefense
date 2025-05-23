#!/bin/bash
#
# Copyright 2025 Johan Rohdin, Lin Zhang, Hieu-Thi Luong (rohdin@fit.vut.cz, partialspoof@gmail.com, contact@hieuthi.com)
#

set -x
. ./path.sh || exit 1

stage=1
stop_stage=7

LPS_dir=/data/deepfake_corpora/LPS
data=data/llamapartialspoof # data folder
data_type="shard"  # shard/raw

expname=ResNet18-TSTP-emb256-fbank80-frms400-aug0-spFalse-saFalse-Softmax-SGD-epoch100
exp_dir=exp/${expname}

config=../../partialspoof/conf/resnet.yaml #wespeaker version 
trained_exp_dir=../../partialspoof/v03_resnet18/exp/${expname}
trained_data=../../partialspoof/v03_resnet18/data/partialspoof

avg_model=$trained_exp_dir/models/avg_model.pt
model_path=$avg_model

gpus="[0]"

. tools/parse_options.sh || exit 1

#######################################################################################
# Stage 1. Preparing data folder for LlamaPartialSpoof: wav.scp, utt2cls, cls2utt, reco2dur
#######################################################################################
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh ${LPS_dir} ${data}
fi

#######################################################################################
# Stage 2. Preapring shard data for partialspoof and musan/rirs 
#######################################################################################
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert evaluation data 0a to ${data_type}..."
  if [ $data_type == "shard" ]; then
      python tools/make_shard_list.py --num_utts_per_shard 1000 \
          --num_threads 8 \
          --prefix shards \
          --shuffle \
          ${data}/0a/wav.scp ${data}/0a/utt2cls \
          ${data}/0a/shards ${data}/0a/shard.list
  else
      python tools/make_raw_list.py --vad_file ${data}/0a/vad ${data}/0a/wav.scp \
          ${data}/0a/utt2cls ${data}/0a/raw.list
  fi
fi


#######################################################################################
# Stage 4. Averaging the model, and extract embeddings
#######################################################################################
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Extract embeddings ..."
  num_gpus=1
  if [[ $(hostname -f) == *fit.vutbr.cz   ]]; then
     gpus=$(python -c "from sys import argv; from safe_gpu import safe_gpu; safe_gpu.claim_gpus(int(argv[1])); print( safe_gpu.gpu_owner.devices_taken )" $num_gpus | sed "s: ::g")
  fi

  wavs_num=$(wc -l ${data}/0a/wav.scp | awk '{print $1}')
  bash tools/extract_embedding.sh --exp_dir ${trained_exp_dir} \
    --model_path $model_path \
    --data_type ${data_type} \
    --data_list ${data}/0a/${data_type}.list \
    --wavs_num ${wavs_num} \
    --store_dir 0a \
    --batch_size 1 \
    --num_workers 1 \
    --nj $num_gpus \
    --gpus $gpus 
  mkdir -p ${exp_dir}/embeddings/
  mv ${trained_exp_dir}/embeddings/0a ${exp_dir}/embeddings

fi

#######################################################################################
# Stage 5. Extract logits and posterior 
#######################################################################################
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Extract logits and posteriors ..."
  mkdir -p ${exp_dir}/posteriors/0a
  echo "LlamaPartialSpoof 0a"
  python wedefense/bin/infer.py --model_path $model_path \
	  --config ${trained_exp_dir}/config.yaml \
	  --num_classes 2 \
	  --embedding_scp_path ${exp_dir}/embeddings/0a/embedding.scp \
	  --out_path ${exp_dir}/posteriors/0a
fi

#######################################################################################
# Stage 6. Convert logits to llr 
#######################################################################################
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Convert logits to llr ..."
  python wedefense/bin/logits_to_llr.py \
	  --logits_scp_path ${exp_dir}/posteriors/0a/logits.scp \
	  --training_counts ${trained_data}/train/cls2num_utts \
	  --train_label ${trained_data}/train/utt2cls \
	  --pi_spoof 0.05

fi

#######################################################################################
# Stage 7. Measuring performance 
#######################################################################################
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Measuring Performance ..."
  # Preparing trails
  # filename        cm-label
  echo "filename cm-label" > ${data}/0a/cm_key_file.txt	  
  cat ${data}/0a/utt2cls >> ${data}/0a/cm_key_file.txt
  sed -i "s/ /\t/g" ${data}/0a/cm_key_file.txt

  echo "Measuring LlamaPartialSpoof 0a"
  python wedefense/metrics/detection/evaluation.py  \
	--m t1 \
	--cm ${exp_dir}/posteriors/0a/llr.txt \
	--cm_key ${data}/0a/cm_key_file.txt
fi

exit 0

