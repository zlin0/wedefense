## 🔎 PartialSpoof Results on ResNet18

- **Input Feature**: `shard` or `raw` waveform
- **Frame Configuration**: (1) 300 frames per segment, 10 ms frame shift, or (2) varied-length input (with max_num_frames=800) 
- **Evaluation Metrics**: minDCF, EER (%), Cllr, actDCF
- **Evaluation Sets**: Dev / Eval

---
### 🧪 Reproducibility Instructions

#### 🔹 Using chunk = 300 frames

```bash
./run.sh --stage 3 --stop_stage 7 \
         --config conf/resnet.yaml \
         --exp_dir exp/ResNet18-TSTP-emb256-fbank80-frms300-aug0-spFalse-saFalse-Softmax-SGD-epoch100
```
#### 🔹Varied-length input (with max_num_frames=800) 

```bash
./run.sh --stage 3 --stop_stage 7 \
         --config conf/resnet_wholeutt_lengthsampler.yaml \
         --exp_dir exp/ResNet18-TSTP-emb256-fbank80-wholeutt-lengthsampleraug0-spFalse-saFalse-Softmax-SGD-epoch100
```
---
### 🧪 Results

| Front-end           | Dev-minDCF | Dev-EER | Dev-Cllr | Dev-actDCF | Eval-minDCF | Eval-EER | Eval-Cllr | Eval-actDCF |
|:--------------------|:----------:|:--------:|:---------:|:------------:|:-------------:|:----------:|:-----------:|:-------------:|
|            |            |         |          |            |             |          |           |             |
|            |            |         |          |            |             |          |           |             |
|            |            |         |          |            |             |          |           |             |
|            |            |         |          |            |             |          |           |             |
|            |            |         |          |            |             |          |           |             |
| **------** |  |  |  |  |  |  |  |  |
|            |            |         |          |            |             |          |           |             |
|            |            |         |          |            |             |          |           |             |
|            |            |         |          |            |             |          |           |             |
|            |            |         |          |            |             |          |           |             |



### Fusion

|      |      | Dev-minDCF | Dev-EER | Dev-Cllr | Dev-actDCF | Eval-minDCF | Eval-EER | Eval-Cllr | Eval-actDCF |
| ---- | ---- | ---------- | ------- | -------- | ---------- | ----------- | -------- | --------- | ----------- |
|      |      |            |         |          |            |             |          |           |             |
|      |      |            |         |          |            |             |          |           |             |



## Citation

If you find xxx useful, please cite it as

```bibtex



```

