# Quick Start

## ASVspoof2019 LA, single GPU
1) Prepare data
- Download ASVspoof2019 LA and set `ASVspoof_dir` to its root.
- From project root:
```bash
cd egs/detection/asvspoof2019/v03_resnet18
bash run.sh --stage 1 --stop_stage 2 --ASVspoof_dir /path/to/ASVspoof2019_LA
```

It will use [egs/detection/asvspoof2019/v03_resnet18/run.sh](egs/detection/asvspoof2019/v03_resnet18/run.sh) stages 1–2 to build lists:
  - `data/asvspoof2019/{train,dev,eval}/wav.scp`
  - `data/asvspoof2019/{train,dev,eval}/utt2lab`
  - `data/asvspoof2019/{train,dev,eval}/raw.list` or `shard.list`
- `local/prepare_data.sh` populates wav.scp/utt2lab; `tools/make_raw_list.py` or `tools/make_shard_list.py` creates the lists consumed by training.
- For other datasets, prepare the same files under `data/<dataset>/<split>/` following the ASVspoof2019 layout.


2) Train a baseline
```bash
bash run.sh --stage 3 --stop_stage 3 --gpus "[0]" --ASVspoof_dir /path/to/ASVspoof2019_LA
```
Expected log snippet (from exp_dir/train.log):
```
INFO exp_dir is: exp/ResNet18-i5p5-smallWeightDecay-earlystop
INFO Train epoch iteration number: 820
INFO Epoch 1/50 step 100 loss=0.48 acc=0.87 lr=1.0e-3
```

3) Evaluate (embeddings -> logits -> llr -> metrics)
```bash
bash run.sh --stage 4 --stop_stage 7 --gpus "[0]" --ASVspoof_dir /path/to/ASVspoof2019_LA
```

