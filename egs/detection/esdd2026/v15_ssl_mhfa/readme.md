## 🔎 ESDD2026 Results

- **Input Feature**: Raw waveform
- **Frame Configuration**: 200 frames per segment
- **Training Strategy**: 8 epochs with full fine-tuning; no data augmentation applied
- **Evaluation Metrics**: minDCF, EER (%), Cllr, actDCF
- **Frontend Setup**: All SSL models are fine-tuned end-to-end (frozen=False)
- **Evaluation Sets**: Dev / Prog / Eval
- **Back-end**: Multi-Head Factorized Attentive Pooling (MHFA)/ MHFA - DSU

---

### 🧪 Reproducibility Instructions

#### 🔹 Model Preparation

**BEATs Model:**
```bash
# Download BEATs models from https://github.com/microsoft/unilm/tree/master/beats
# Place the model file to ./beats_models_cache/BEATs_iter3.pt
```

**EAT Models:**
```bash
# EAT models will be automatically downloaded from HuggingFace
# Cache location: ./eat_models_cache/
# - worstchan/EAT-base_epoch30_pretrain (for baseline)
# - worstchan/EAT-base_epoch30_finetune_AS2M (for DSU variant)
```

**Dasheng Models:**
```bash
# Dasheng models will be automatically downloaded from HuggingFace
# Cache location: ./dasheng_models_cache/
# - mispeech/dasheng-0.6B
# - mispeech/dasheng-1.2B
```

#### 🔹 Training Examples

**BEATs with MHFA:**
```bash
./run.sh --stage 3 --stop_stage 7 \
         --config conf/beats/mhfa_baseline.yaml \
         --exp_dir exp/BEATs_MHFA-emb256-beats-num_frms200-aug0-spFalse-saFalse-Softmax-AdamW-epoch8
```

---

### 🧪 SSL-based Models (Full Fine-tuning)

Performance comparison on ESDD 2026. \* denotes the system uses MHFA-DSU.

| ID | System | Dev-EER (%) | Prog-EER (%) | Eval-EER (%) |
|:---|:-------|:-----------:|:------------:|:------------:|
| 1  | BEATs | 0.00 | 7.10 | - |
| 2  | Dasheng-0.6B | 0.27 | - | - |
| 3  | Dasheng-1.2B | 0.33 | - | - |
| 4 | EAT Base | 0.00 | 5.41 | - |
| 5 | EAT Base_FT(AS2M)\* | 0.00 | 4.77 | 4.80 |

**Note:** Results are reported as EER (%). The EAT Base_FT(AS2M)* system uses MHFA-DSU backend.

---

### 📋 Configuration Details

**Common Training Settings:**
- **Epochs**: 8
- **Batch Size**: 128
- **Optimizer**: AdamW (weight_decay=1.0e-4)
- **Learning Rate**: Exponential decrease from 5.0e-4 to 1.0e-5 (warm-up: 2 epochs)
- **Frontend LR Ratio**: 0.05
- **Frame Length**: 200 frames
- **Resample Rate**: 16 kHz
- **Data Augmentation**: None (aug_prob=0.0, speed_perturb=False, spec_aug=False)

**Model Architecture:**
- **Head Number**: 32
- **Embedding Dimension**: 256
- **Compression Dimension**: 128

**Dataset Configuration:**
- **Frame Filtering**: 50-400 frames
- **CMVN**: 
  - BEATs/Dasheng: False
  - EAT: True (norm_mean=True, norm_var=False)

**Special Configurations:**
- **EAT-base-ft with DSU**: 
  - Model: SSL_BACKEND_MHFA_DSU
  - Augmentation Mode: DSU (aug_prob=0.5, dsu_factor=1.0, beta_alpha=0.3)

---

### 🔧 Run Script Stages

The `run.sh` script supports the following stages:

1. **Stage 1**: Prepare datasets (wav.scp, utt2lab, lab2utt, reco2dur)
2. **Stage 2**: Convert train/test data to raw format
3. **Stage 3**: Training
4. **Stage 4**: Model averaging and embedding extraction
5. **Stage 5**: Extract logits and posteriors
6. **Stage 6**: Convert logits to LLR (Log-Likelihood Ratio)
7. **Stage 7**: Performance evaluation (minDCF, EER, Cllr, actDCF)

---

### 📝 Notes

- **Full Fine-tuning**: All SSL models are fine-tuned end-to-end (frozen=False)
- **Model Averaging**: By default, the last 2 checkpoints are averaged (num_avg=2)

---

## Citation

If you find MHFA useful, please cite it as

```bibtex
@inproceedings{peng2023attention,
  title={An attention-based backend allowing efficient fine-tuning of transformer models for speaker verification},
  author={Peng, Junyi and Plchot, Old{\v{r}}ich and Stafylakis, Themos and Mo{\v{s}}ner, Ladislav and Burget, Luk{\'a}{\v{s}} and {\v{C}}ernock{\`y}, Jan},
  booktitle={2022 IEEE Spoken Language Technology Workshop (SLT)},
  pages={555--562},
  year={2023},
  organization={IEEE}
}
```
