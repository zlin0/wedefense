#!/bin/bash

# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#           2025 Johan Rohdin, Lin Zhang (rohdin@fit.vut.cz, partialspoof@gmail.com)
#           2025 Junyi Peng (pengjy@fit.vut.cz)

set -x
. ./path.sh || exit 1

stage=3
stop_stage=3

LAVDF_dir=/export/fs05/arts/dataset/LAV-DF/LAV-DF/
data=data/lav-df # data folder
data_type="raw"  # shard/raw

config=conf/MHFA_wav2vec2_xlsr53-FT-1stage.yaml
exp_dir=exp/MHFA_wav2vec2_xlsr53-FT-1stage
gpus="[0]"
num_avg=2 # how many models you want to average
checkpoint=
score_norm_method="asnorm"  # asnorm/snorm
top_n=300
DSETs="train dev test"

# resolution for evaluation
eval_reso=10

# Not applied. # setup for large margin fine-tuning.
# lm_config=conf/campplus_lm.yaml

. tools/parse_options.sh || exit 1

#######################################################################################
# Stage 1. Preparing data folder for database: wav.scp, utt2lab, lab2utt, reco2dur, rttm
#######################################################################################
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  python ./local/json_to_rttm_kaldi.py \
	 --json_file ${LAVDF_dir}/metadata.json \
	 --save_dir ${data}
  for dset in ${DSETs}; do
     # create lab2utt
      ./tools/utt2lab_to_lab2utt.pl ${data}/${dset}/utt2lab >${data}/${dset}/lab2utt
  done
  echo "Prepared data for lavdf"
fi

#######################################################################################
# Stage 2. Preapring shard data for database and musan/rirs
#######################################################################################
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  # We don't use VAD here
  for dset in ${DSETs} ;do
      if [ $data_type == "shard" ]; then
          python tools/make_shard_list.py --num_utts_per_shard 1000 \
              --num_threads 8 \
              --prefix shards \
              --shuffle \
              ${data}/$dset/wav.scp ${data}/$dset/utt2lab \
              ${data}/$dset/shards ${data}/$dset/shard.list
      else
          python tools/make_raw_list.py --vad_file ${data}/$dset/vad ${data}/$dset/wav.scp \
              ${data}/$dset/utt2lab ${data}/$dset/raw.list
      fi
  done

fi

#######################################################################################
# Stage 3. Training
#######################################################################################
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=1
  if [[ $(hostname -f) == *fit.vutbr.cz   ]]; then
     gpus=$(python -c "from sys import argv; from safe_gpu import safe_gpu; safe_gpu.claim_gpus(int(argv[1])); print( safe_gpu.gpu_owner.devices_taken )" $num_gpus | sed "s: ::g")
  fi
    ##num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
    #python -m pdb \
    torchrun --rdzv_backend=c10d --rdzv_endpoint=$(hostname):$((RANDOM)) --nnodes=1 --nproc_per_node=$num_gpus \
      wedefense/bin/train_localization.py --config $config \
        --exp_dir ${exp_dir} \
        --gpus $gpus \
        --num_avg ${num_avg} \
        --data_type "${data_type}" \
        --train_data ${data}/train/${data_type}.list \
        --train_label ${data}/train/rttm \
        ${checkpoint:+--checkpoint $checkpoint}
        # Note, label was assigned in stage 2,
	# utt2lab here only for label2id.
        #--reverb_data data/rirs/lmdb \
        #--noise_data data/musan/lmdb \
	#TODO, currently also moved from local/extract_emb.sh, flexible to control musan/rirs.
fi || exit

avg_model=$exp_dir/models/avg_model.pt
model_path=${checkpoint:-$avg_model}
#######################################################################################
# Stage 4. Averaging the model, and extract embeddings
#######################################################################################
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  python wedefense/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}


  echo "Extract embeddings ..."
  num_gpus=1
  if [[ $(hostname -f) == *fit.vutbr.cz   ]]; then
     gpus=$(python -c "from sys import argv; from safe_gpu import safe_gpu; safe_gpu.claim_gpus(int(argv[1])); print( safe_gpu.gpu_owner.devices_taken )" $num_gpus | sed "s: ::g")
  fi

  #TODO
  #1. currently too much layers to call the extracting script.
  #2. not friend for slurm when spliting list
  local/extract_emb.sh \
     --exp_dir $exp_dir --model_path $model_path \
     --nj $num_gpus --gpus $gpus --data_type $data_type --data ${data}
fi

#######################################################################################
# Stage 5. Extract logits and posterior
#######################################################################################
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Extract logits and posteriors ..."
  for dset in ${DSETs};do
      mkdir -p ${exp_dir}/posteriors/$dset
      echo $dset
      python wedefense/bin/infer_by_utt.py --model_path $model_path \
	  --config ${exp_dir}/config.yaml \
	  --num_classes 2 \
	  --embedding_scp_path ${exp_dir}/embeddings/$dset/embedding.scp \
	  --out_path ${exp_dir}/posteriors/$dset
  done
fi


#######################################################################################
# Stage 6. Print logits to txt
#######################################################################################
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Convert logits to llr ..."
  cut -f2 -d" " ${data}/train/utt2lab | sort | uniq -c | awk '{print $2 " " $1}' > ${data}/train/lab2num_utts
  for dset in ${DSETs};do
      echo $dset
      python wedefense/utils/print_frame_logits.py \
        --logits_scp_path ${exp_dir}/posteriors/$dset/logits.scp \
        --score_reso 20 \
        --eval_reso ${eval_reso} \
        --train_label ${data}/train/rttm \
        --eval_label ${data}/$dset/rttm
      #comment out the last row for eval_label if you don't have the ground truth.

      # TODO
      python -m pdb wedefense/utils/diarization/convert_frame_score_to_rttm.py \
	  --logits_scp_path ${exp_dir}/posteriors/$dset/logits.scp \
	  --score_reso 20 \
	  --output_rttm ${exp_dir}/posteriors/${dset}/logits_rttm.txt \
	  --frame_index True --label_exist True

  done
fi

#######################################################################################
# Stage 7. Measuring performance
#######################################################################################
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Measuring Performance ..."
  for dset in ${DSETs};do
    # Preparing trails

    echo "Measuring " $dset
    python wedefense/metrics/localization/point_eer.py  \
	--score_file ${exp_dir}/posteriors/${dset}/logits_frame_${eval_reso}ms.txt \
	--score_reso ${eval_reso}

    #TODO Unify variable names and usage
    frame_dur=$(echo "scale=3; ${eval_reso} / 1000" | bc)  #convert ms to sec.
    python wedefense/metrics/localization/rangeeer.py  \
	--score_file ${exp_dir}/posteriors/${dset}/logits_frame_${eval_reso}ms.txt
	--score_index 3 \
	--rttm_file ${data}/$dset/rttm \
	--frame_duration ${frame_dur}
  done
fi

#######################################################################################
# Stage 8. Analyses
#######################################################################################
# TODO
# 1. significant test
# 2. boostrap testing
# 3. embedding visulization
exit 0

