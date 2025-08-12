## 🔎 Hybrid Pruning for Anti-Spoofing Results

- **Input Feature**: Raw waveform (via SSL model)
- **Frame Configuration**: 150 frames per segment, 20 ms frame shift
- **Training Strategy**: Jointly optimizing for task performance and model sparsity in a single stage. A warm-up schedule is used where the sparsity target linearly increases from 0 to the final value over the first 5 epochs.
- **Evaluation Metrics**: minDCF, EER (%)
- **Evaluation Sets**: Dev / Eval
- **Back-end**: Multi-Head Factorized Attentive Pooling (MHFA)

---

### **Checkpoint**

This work use a pre-trained SSL model (i.e., **WavLM Base**), which has achieved SOTA performance on the ASVspoof 5 dataset.

You can access the official model checkpoint and usage details on the Hugging Face Hub:

- **Model Link**: [JYP2024/Wedefense\_ASV2025\_WavLM\_Base\_Pruning](https://huggingface.co/JYP2024/Wedefense_ASV2025_WavLM_Base_Pruning)

---

### **Results on ASVspoof 5**

The following table compares the performance of our proposed **Hybrid Pruning (HP) single system** against other top-performing systems from the official ASVspoof 5 Challenge leaderboard.

| System | Dev minDCF | Dev EER (%) | Eval minDCF | Eval EER (%) |
| :--- | :--- | :--- | :--- | :--- |
| Rank 3 (ID:T27, Fusion) | - | - | 0.0937 | 3.42 |
| **HP (ours, Single system)** | 0.0395 | 1.547 | **0.1028** | **3.758** |
| Rank 4 (ID:T23, Fusion) | - | - | 0.1124 | 4.16 |
| Rank 9 (ID:T23, Best single system) | - | - | 0.1499 | 5.56 |
