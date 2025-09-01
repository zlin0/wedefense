#!/bin/bash
#
# Copyright 2025 Johan Rohdin, Lin Zhang (rohdin@fit.vut.cz, partialspoof@gmail.com)
#

set -e -o pipefail -x
. ./path.sh || exit 1

stage=1
stop_stage=7

# TODO Please modify the following paths to your own data directories
ASVspoof_dir=/export/fs05/arts/dataset/ASVspoof5
MUSAN_dir=/path/to/your/musan # e.g., /export/fs05/arts/dataset/musan
RIRS_dir=/path/to/your/rirs # e.g., /share/workspace/shared_datasets/speechdata/21_RIRS_NOISES/RIRS_NOISES
data=data/asvspoof5 # data folder
data_type="shard"  # shard/raw

config=conf/resnet.yaml
exp_dir=exp/ResNet18-TSTP-emb256-fbank80-frms400-aug0-spFalse-saFalse-Softmax-SGD-epoch100

gpus="[0]" # Specify GPUs to use, e.g., "[0]" or "[0,1]"
num_avg=2 # Number of models to average.
          # Set to > 0 to activate model averaging and use the averaged model.
          # Set to 0 or a negative value to use the single best_model.pt.
save_epoch_interval=5 # Save model each N epoch
early_stop_patience=-1 # Early stop based on the performance on Val, <0 for no early stop.
validate_interval=1 # Evaluate validation set every N epochs.
checkpoint=

. tools/parse_options.sh || exit 1

# Count the number of GPUs, handling potential spaces
num_gpus=$(echo "$gpus" | tr -d '[] ' | awk -F',' '{print NF}')

#######################################################################################
# Stage 1. Preparing data folder for ASVspoof5: wav.scp, utt2lab, lab2utt, reco2dur
#######################################################################################
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh "${ASVspoof_dir}" "${data}"
fi

#######################################################################################
# Stage 2. Preparing shard data for ASVspoof5 and musan/rirs
#######################################################################################
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  # We don't use VAD here
  for dset in train dev eval;do
      if [ "${data_type}" == "shard" ]; then
          python tools/make_shard_list.py --num_utts_per_shard 1000 \
              --num_threads 8 \
              --prefix shards \
              --shuffle \
              "${data}/${dset}/wav.scp" "${data}/${dset}/utt2lab" \
              "${data}/${dset}/shards" "${data}/${dset}/shard.list"
      else
          python tools/make_raw_list.py \
            "${data}/${dset}/wav.scp" \
                  "${data}/${dset}/utt2lab" "${data}/${dset}/raw.list"
      fi
  done

  echo "Preparing augmentation data (MUSAN and RIRS)..."
  # Check if the augmentation data directories are provided
  if [ -d "${MUSAN_dir}" ] && [ -d "${RIRS_dir}" ]; then
    # Create wav.scp for MUSAN and RIRS
    find "${MUSAN_dir}" -name "*.wav" | awk -F"/" '{print $NF,$0}' | sort > data/musan/wav.scp
    find "${RIRS_dir}" -name "*.wav" | awk -F"/" '{print $NF,$0}' | sort > data/rirs/wav.scp

    # Convert augmentation data to LMDB in parallel for speed.
    # NOTE: LMDB may not work on NFS. We create it locally and then rsync.
    (
      python tools/make_lmdb.py data/musan/wav.scp "${HOME}/local_lmdb/musan/lmdb" && \
      rsync -av "${HOME}/local_lmdb/musan/lmdb" data/musan/lmdb
    ) &
    (
      python tools/make_lmdb.py data/rirs/wav.scp "${HOME}/local_lmdb/rirs/lmdb" && \
      rsync -av "${HOME}/local_lmdb/rirs/lmdb" data/rirs/lmdb
    ) &
    wait
  else
    echo "Warning: MUSAN_dir or RIRS_dir not set. Skipping augmentation data preparation."
  fi
fi

#######################################################################################
# Stage 3. Training
#######################################################################################
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  if [[ $(hostname -f) == *fit.vutbr.cz   ]]; then
     gpus=$(python -c "from sys import argv; from safe_gpu import safe_gpu; safe_gpu.claim_gpus(int(argv[1])); print( safe_gpu.gpu_owner.devices_taken )" $num_gpus | sed "s: ::g")
  fi

  # Use a random port for torchrun from the dynamic port range
  torchrun --rdzv_backend=c10d --rdzv_endpoint=$(hostname):$((RANDOM % 16384 + 49152)) --nnodes=1 --nproc_per_node=$num_gpus \
      wedefense/bin/train.py --config "$config" \
        --exp_dir "${exp_dir}" \
        --gpus "$gpus" \
        --num_avg ${num_avg} \
        --data_type "${data_type}" \
        --train_data "${data}/train/${data_type}.list" \
        --train_label "${data}/train/utt2lab" \
        --val_data "${data}/dev/${data_type}.list" \
        --val_label "${data}/dev/utt2lab" \
        --early_stop_patience ${early_stop_patience} \
        --validate_interval ${validate_interval} \
        ${checkpoint:+--checkpoint "$checkpoint"}
        #--reverb_data data/rirs/lmdb \
        #--noise_data data/musan/lmdb \
	#TODO, currently also moved from local/extract_emb.sh, flexible to control musan/rirs.
fi

avg_model=$exp_dir/models/avg_model.pt
best_model=$exp_dir/models/best_model.pt

if [ ${num_avg} -gt 0 ]; then
  model_path=$avg_model
else
  model_path=$best_model
fi
#######################################################################################
# Stage 4. Averaging the model, and extract embeddings
#######################################################################################
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  if [ ${num_avg} -gt 0 ]; then
    echo "Do model average ..."
    python wedefense/bin/average_model.py \
      --dst_model "$avg_model" --src_path "${exp_dir}/models" --num ${num_avg}
  fi

  echo "Extract embeddings ..."
  if [[ $(hostname -f) == *fit.vutbr.cz   ]]; then
     gpus=$(python -c "from sys import argv; from safe_gpu import safe_gpu; safe_gpu.claim_gpus(int(argv[1])); print( safe_gpu.gpu_owner.devices_taken )" "$num_gpus" | sed "s: ::g")
  fi

  local/extract_emb.sh \
     --exp_dir "$exp_dir" --model_path "$model_path" \
     --nj "$num_gpus" --gpus "$gpus" --data_type "$data_type" --data "${data}"
fi

#######################################################################################
# Stage 5. Extract logits and posterior
#######################################################################################
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Extract logits and posteriors ..."
  for dset in dev eval;do
      mkdir -p "${exp_dir}/posteriors/${dset}"
      echo "Processing ${dset} set..."
      python wedefense/bin/infer.py --model_path "$model_path" \
        --config "${exp_dir}/config.yaml" \
        --num_classes 2 \
        --embedding_scp_path "${exp_dir}/embeddings/${dset}/embedding.scp" \
        --out_path "${exp_dir}/posteriors/${dset}"
  done
fi

#######################################################################################
# Stage 6. Convert logits to llr
#######################################################################################
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Convert logits to llr ..."
  cut -f2 -d" " "${data}/train/utt2lab" | sort | uniq -c | awk '{print $2 " " $1}' > "${data}/train/lab2num_utts"
  for dset in dev eval; do
      echo "Processing ${dset} set..."
      python wedefense/bin/logits_to_llr.py \
        --logits_scp_path "${exp_dir}/posteriors/${dset}/logits.scp" \
        --training_counts "${data}/train/lab2num_utts" \
        --train_label "${data}/train/utt2lab" \
        --pi_spoof 0.05

  done
fi

#######################################################################################
# Stage 7. Measuring performance
#######################################################################################
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Measuring Performance ..."
  for dset in dev eval; do
    # Preparing trails
    # filename        cm-label
    echo "filename cm-label" > "${data}/${dset}/cm_key_file.txt"
    cat "${data}/${dset}/utt2lab" >> "${data}/${dset}/cm_key_file.txt"
    sed -i "s/ /\t/g" "${data}/${dset}/cm_key_file.txt"

    echo "Measuring ${dset}..."
    python wedefense/metrics/detection/evaluation.py  \
      --m t1 \
      --cm "${exp_dir}/posteriors/${dset}/llr.txt" \
            --cm_key "${data}/${dset}/cm_key_file.txt" 2>&1 | tee "${exp_dir}/results_${dset}.txt"
  done
fi

#######################################################################################
# Stage 8. Analyses
#######################################################################################
# TODO
# 1. significant test
# 2. boostrap testing

exit 0
