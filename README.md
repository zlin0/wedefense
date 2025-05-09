# wedefense


## Enviorment:
```shell
conda create -n wedefense_new python=3.10
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```
If you got warn: ModuleNotFoundError: No module named 'whisper'
```shell
pip install -U openai-whisper --no-cache-dir
```
If you are working on merlin:
```shell
pip install safe_gpu
```


## Updates
2025-05-08: [Lin] clean code for stage 5-7 of `egs/detection/partialspoof/v03_resnet18/`
2025-05-07: [Lin] clean code for stage 1-4 of `egs/detection/partialspoof/v03_resnet18/`


## TODO:

[Lin]: 
1. add varied-length input, 
2. add augmentation: codec, mixup, etc.
3. add new models for detection: LCNN, SSL

[Junyi]:
1. SSL supporting for 

[Lin, Shuai] move wespeaker part to submodule







Reference:
1. Mainly adapated from [wespeaker](https://github.com/wenet-e2e/wespeaker)
