## 🔎 PartialSpoof Results

- **Input Feature**: Raw waveform
- **Frame Configuration**: 150 frames per segment, 20 ms frame shift
- **Training Strategy**: 20 epochs for frozen front-ends, 5 epochs for full fine-tuning; no data augmentation applied
- **Evaluation Metrics**: minDCF, EER (%), Cllr, actDCF
- **Frontend Setup**: All SSL models are frozen unless explicitly specified
- **Evaluation Sets**: Dev / Eval
- **Back-end**: Multi-Head Factorized Attentive Pooling (MHFA)

---

### 🧪 SSL-based models (Frozen)

| Front-end           | Dev-minDCF | Dev-EER | Dev-Cllr | Dev-actDCF | Eval-minDCF | Eval-EER | Eval-Cllr | Eval-actDCF |
|:--------------------|:----------:|:--------:|:---------:|:------------:|:-------------:|:----------:|:-----------:|:-------------:|
| wav2vec2-base | 0.05 | 1.93 | 0.09 | 0.05 | 0.09 | 3.83 | 0.23 | 0.10 |
| HuBERT-base         | 0.05 | 2.28 | 0.09 | 0.05 | 0.12 | 4.85 | 0.33 | 0.14 |
| WavLM-base+         | 0.04 | 1.56 | 0.12 | 0.06 | 0.06 | 2.68 | 0.31 | 0.13 |
| WavLM-base          | 0.04 | 1.65 | 0.07 | 0.04 | 0.08 | 3.40 | 0.23 | 0.10 |
| Data2Vec-base       | 0.10 | 3.76 | 0.18 | 0.11 | 0.15 | 5.68 | 0.33 | 0.16 |
| **--- Base / Large Split ---** |  |  |  |  |  |  |  |  |
| wav2vec2-large      | 0.05 | 2.15 | 0.10 | 0.07 | 0.09 | 3.84 | 0.22 | 0.13 |
| WavLM-large         | 0.01 | 0.46 | 0.03 | 0.01 | 0.04 | 1.38 | 0.09 | 0.04 |
| Data2Vec-large      | 0.04 | 1.57 | 0.12 | 0.06 | 0.06 | 2.61 | 0.28 | 0.12 |
| XLSR-53             | 0.05 | 2.16 | 0.09 | 0.06 | 0.08 | 3.63 | 0.19 | 0.11 |
| HuBERT-large        | 0.03 | 1.21 | 0.05 | 0.03 | 0.05 | 2.23 | 0.15 | 0.08 |

---

### 🧪 SSL  (Full fine-tuning)

---

| Front-end | Training | Dev-minDCF | Dev-EER | Dev-Cllr | Dev-actDCF | Eval-minDCF | Eval-EER | Eval-Cllr | Eval-actDCF |
|:----------|:---------|:----------:|:--------:|:---------:|:------------:|:-------------:|:----------:|:-----------:|:-------------:|
| WavLM-base+   | Frozen      | 0.04 | 1.56 | 0.12 | 0.06 | 0.06 | 2.68 | 0.31 | 0.13 |
| WavLM-base+ | Full Finetuning | 0.02 | 0.71 | 0.05 | 0.03 | 0.04 | 1.59 | 0.20 | 0.08 |

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


