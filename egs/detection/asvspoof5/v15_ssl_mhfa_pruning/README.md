## 🔎 Hybrid Pruning for Anti-Spoofing Results

This repository implements the **Hybrid Pruning** framework from the paper ["Hybrid Pruning: In-Situ Compression of Self-Supervised Speech Models for Speaker Verification and Anti-Spoofing"](https://arxiv.org/abs/2508.16232) (arXiv:2508.16232).

- **Input Feature**: Raw waveform (via SSL model)
- **Frame Configuration**: 150 frames per segment, 20 ms frame shift
- **Training Strategy**: Jointly optimizing for task performance and model sparsity in a single stage. A progressive warm-up schedule is used where the sparsity target increases from 0 to the final value over the first 5 epochs using configurable scheduling strategies.
- **Evaluation Metrics**: minDCF, EER (%)
- **Evaluation Sets**: Dev / Eval
- **Back-end**: Multi-Head Factorized Attentive Pooling (MHFA)

### **Key Features**

- **Progressive Pruning Strategy**: Supports multiple sparsity scheduling strategies (linear, cosine, exponential, sigmoid) for gradual model compression
- **Structured Pruning**: Implements Hard Concrete distribution for differentiable structured pruning of channels, attention heads, and layers
- **Dual-Formulation Optimization**: Uses separate optimizers for main model parameters and pruning regularization parameters
- **Multi-Granularity Pruning**: Prunes at different levels (convolutional channels, attention heads, feed-forward layers)
- **Lagrangian Constraint**: Enforces target sparsity using quadratic penalty functions with learnable Lagrange multipliers

---

### **Pruning Configuration**

The pruning system supports flexible configuration through YAML files. Key parameters include:

```yaml
# Pruning configuration
use_pruning_loss: True                    # Enable pruning loss
target_sparsity: 0.1                     # Final sparsity level (10% parameters remaining)
initial_reg_lr: 5.0e-2                   # Learning rate for regularization parameters
sparsity_warmup_epochs: 5                # Number of warmup epochs
sparsity_schedule: "cosine"              # Progressive schedule type
min_sparsity: 0.0                        # Initial sparsity level
```

**Supported Schedule Types:**
- `linear`: Linear increase in sparsity
- `cosine`: Cosine annealing schedule (smooth start and end)
- `exponential`: Exponential increase (slow start, fast end)
- `sigmoid`: Sigmoid schedule (very gradual start and end)

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

**Key Achievements:**
- **State-of-the-art performance**: 3.7% EER on ASVspoof5 evaluation set
- **Significant compression**: Up to 70% parameter reduction with negligible performance degradation
- **Improved generalization**: Better performance in low-resource scenarios, reducing overfitting

---

### **Citation**

If you use this work, please cite the original paper:

```bibtex
@article{peng2025hybrid,
  title={Hybrid Pruning: In-Situ Compression of Self-Supervised Speech Models for Speaker Verification and Anti-Spoofing},
  author={Peng, Junyi and Zhang, Lin and Han, Jiangyu and Plchot, Oldřich and Rohdin, Johan and Stafylakis, Themos and Wang, Shuai and Černocký, Jan},
  journal={arXiv preprint arXiv:2508.16232},
  year={2025}
}
```

**Paper Link**: [https://arxiv.org/abs/2508.16232](https://arxiv.org/abs/2508.16232)

---

### **Additional Results**

Based on the paper, our Hybrid Pruning framework also achieves excellent results on speaker verification tasks:

- **VoxCeleb1-O**: 0.7% EER
- **VoxCeleb1-E**: 0.8% EER
- **VoxCeleb1-H**: 1.6% EER

