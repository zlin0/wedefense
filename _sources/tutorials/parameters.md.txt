
# Parameters with WeDefense
**Author:** Lin Zhang
**Date:** 2025-10-06
**Status:** working on


Here we will introduce parameters based on `egs/partialspoof/v03_resnet18/conf/resnet.yaml` as an example to introduce parameters. 
```yaml
exp_dir: exp/ResNet18-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch100
gpus: "[0,1]"
num_avg: 10
enable_amp: False # whether enable automatic mixed precision training
```

