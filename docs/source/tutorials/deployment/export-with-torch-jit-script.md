# Export model with torch.jit.script()

In this section, we describe how to export a model trained on wedefense via `torch.jit.script()`.

## Overview

The JIT export functionality in wedefense exports the **backend + projection** model (without frontend). The exported model accepts pre-extracted features as input, making it suitable for deployment scenarios where feature extraction is handled separately.

## Prerequisites

- A trained model checkpoint (`.pt` file)
- The corresponding configuration file (`.yaml`) used during training
- PyTorch installed

## Usage

### Basic Command

```bash
python wedefense/bin/export_jit.py \
    --config <path_to_config.yaml> \
    --checkpoint <path_to_checkpoint.pt> \
    --output_file <path_to_output.zip>
```

### Arguments

- `--config` (required): Path to the configuration YAML file used during training
- `--checkpoint` (required): Path to the trained model checkpoint file
- `--output_file` (optional): Path where the exported JIT model will be saved. If not specified, the model will be prepared but not saved.

## Examples

### Example 1: Export Detection Model

```bash
python wedefense/bin/export_jit.py \
    --config pretrain_models/detection_MHFA_wav2vec2_large/config.yaml \
    --checkpoint pretrain_models/detection_MHFA_wav2vec2_large/avg_model.pt \
    --output_file pretrain_models/detection_MHFA_wav2vec2_large/final.zip
```

### Example 2: Export Localization Model

```bash
python wedefense/bin/export_jit.py \
    --config pretrain_models/localization_MFHA_xlsr/config.yaml \
    --checkpoint pretrain_models/localization_MFHA_xlsr/avg_model.pt \
    --output_file pretrain_models/localization_MFHA_xlsr/final.zip
```

## Loading and Using the Exported Model

### Python Example

```python
import torch

# Load the exported JIT model
jit_model = torch.jit.load("pretrain_models/detection_MHFA_wav2vec2_large/final.zip")
jit_model.eval()

# Prepare input features
# Shape: [Batch, Dim, Frame_len, Nb_Layer]
# For example, with batch_size=1, dim=1024, frame_len=100, nb_layer=13
features = torch.randn(1, 1024, 100, 13)

# Run inference
with torch.no_grad():
    embeddings = jit_model(features)

print(f"Output shape: {embeddings.shape}")
```

### C++ Example (LibTorch)

```cpp
#include <torch/script.h>
#include <torch/torch.h>

// Load the model
torch::jit::script::Module module;
module = torch::jit::load("pretrain_models/detection_MHFA_wav2vec2_large/final.zip");
module.eval();

// Prepare input features
// Shape: [Batch, Dim, Frame_len, Nb_Layer]
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::randn({1, 1024, 100, 13}));

// Run inference
at::Tensor output = module.forward(inputs).toTensor();
std::cout << "Output shape: " << output.sizes() << std::endl;
```
