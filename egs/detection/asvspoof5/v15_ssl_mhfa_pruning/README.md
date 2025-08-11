## 🔎  Hybrid Pruning for Anti-Spoofing Results

- **Input Feature**: Raw waveform
- **Frame Configuration**: 150 frames per segment, 20 ms frame shift
- **Training Strategy**: Jointly optimizing for task performance and model sparsity in a single stage. A warm-up schedule is used where the sparsity target linearly increases from 0 to the final value over the first 5 epochs.
- **Evaluation Metrics**: minDCF, EER (%), Cllr, actDCF
- **Evaluation Sets**: Dev / Eval
- **Back-end**: Multi-Head Factorized Attentive Pooling (MHFA)