# Export model with torch.jit.script()

In this section, we describe how to export a model trained on wedefense via `torch.jit.script()`.

## Overview

The JIT export functionality in wedefense exports the **backend + projection** model and (when configured) a **frontend** model separately:

- **Backend + Projection**: Accepts pre-extracted features and outputs classification results
- **Frontend**: Accepts raw waveforms and outputs multi-layer features


## Prerequisites

- A trained model checkpoint (`.pt` file)
- The corresponding configuration file (`.yaml`) used during training
- PyTorch installed

## Usage

### Basic Command

```bash
python wedefense/deploy/export_jit.py \
    --config <path_to_config.yaml> \
    --checkpoint <path_to_checkpoint.pt> \
    --output_file <path_to_output.zip>
```

### Arguments

- `--config` (required): Path to the configuration YAML file used during training
- `--checkpoint` (required): Path to the trained model checkpoint file
- `--output_file` (optional): Path where the exported JIT model will be saved. If not specified, the model will be prepared but not saved.

### Output Files

- **Backend + Projection**: `<output_file>.zip` (e.g., `final.zip`) - Contains the backend model and projection layer
- **Frontend** (if applicable): `<output_file_base>_frontend.pt` (e.g., `final_frontend.pt`) - Contains the S3PRL frontend for feature extraction

Note: For S3PRL frontends (wav2vec2_large_960, xlsr_53), a JIT-compatible frontend will be automatically exported.

## Examples

### Example 1: Export Detection Model (wav2vec2_large_960)

```bash
python wedefense/deploy/export_jit.py \
    --config pretrain_models/detection_MHFA_wav2vec2_large/config.yaml \
    --checkpoint pretrain_models/detection_MHFA_wav2vec2_large/avg_model.pt \
    --output_file pretrain_models/detection_MHFA_wav2vec2_large/final.zip
```

This will generate:
- `final.zip` - Backend + Projection model
- `final_frontend.pt` - Frontend model (wav2vec2_large_960)

### Example 2: Export Localization Model (XLSR)

```bash
python wedefense/deploy/export_jit.py \
    --config pretrain_models/localization_MFHA_xlsr/config.yaml \
    --checkpoint pretrain_models/localization_MFHA_xlsr/avg_model.pt \
    --output_file pretrain_models/localization_MFHA_xlsr/final.zip
```

This will generate:
- `final.zip` - Backend + Projection model
- `final_frontend.pt` - Frontend model (XLSR)

## Loading and Using the Exported Model

### Python Example (Backend Only)

```python
import torch

# Load the exported JIT backend model
backend = torch.jit.load("pretrain_models/detection_MHFA_wav2vec2_large/final.zip")
backend.eval()

# Prepare input features
# Shape: [Batch, Dim, Frame_len, Nb_Layer]
# For wav2vec2_large_960: [B, 1024, T, 25]
# For XLSR: [B, 1024, T, 25]
features = torch.randn(1, 1024, 50, 25)

# Run inference
with torch.no_grad():
    output = backend(features)

print(f"Output shape: {output.shape}")  # [1, num_classes]
print(f"Predictions: {output}")
```

### C++ Example (LibTorch, Backend Only)

```cpp
#include <torch/script.h>
#include <torch/torch.h>

int main() {
    // Load the backend model
    torch::jit::script::Module backend;
    backend = torch::jit::load("pretrain_models/detection_MHFA_wav2vec2_large/final.zip");
    backend.eval();

    // Prepare input features: [Batch, Dim, Frame_len, Nb_Layer]
    // For wav2vec2_large_960 and XLSR: [B, 1024, T, 25]
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::randn({1, 1024, 50, 25}));

    // Run inference
    at::Tensor output = backend.forward(inputs).toTensor();
    std::cout << "Output shape: " << output.sizes() << std::endl;
    std::cout << "Predictions: " << output << std::endl;

    return 0;
}
```

Compile with:
```bash
g++ -std=c++14 inference.cpp -o inference \
    -I/path/to/libtorch/include \
    -L/path/to/libtorch/lib \
    -ltorch -ltorch_cpu -lc10
```

### Python Example (Frontend + Backend - Full Pipeline)

```python
import torch

# Load both frontend and backend models
frontend = torch.jit.load("pretrain_models/detection_MHFA_wav2vec2_large/final_frontend.pt")
backend = torch.jit.load("pretrain_models/detection_MHFA_wav2vec2_large/final.zip")
frontend.eval()
backend.eval()

# Prepare raw waveform input: [Batch, Time]
# Example: 1 second of audio at 16kHz
wav = torch.randn(1, 16000)
wav_lengths = torch.tensor([16000], dtype=torch.long)

# Run inference
with torch.no_grad():
    # Frontend: raw audio -> multi-layer features
    # Output shape: [B, D, T, L] where D=1024, L=25 (num_layers)
    features, feat_lengths = frontend(wav, wav_lengths)
    print(f"Frontend output: {features.shape}")  # [1, 1024, ~50, 25]

    # Backend + Projection: features -> classification
    output = backend(features)
    print(f"Backend output: {output.shape}")  # [1, num_classes]
    print(f"Predictions: {output}")
```
