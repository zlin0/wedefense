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

* `egs/detection/partialspoof/v03_resnet18` is relatively complete. I'm still working on updating dataset.py, but you can run it to get familiar with the toolkit.
* 2025-05-08: [Lin] clean code for stage 5-7 of `egs/detection/partialspoof/v03_resnet18/`
* 2025-05-07: [Lin] clean code for stage 1-4 of `egs/detection/partialspoof/v03_resnet18/`



## Folder

```shell
.
├── egs #example folders to include supported tasks/databases.
│   ├── detection
│   │   ├── asvspoof5
│   │   ├── llamapartialspoof
│   │   ├── partialedit
│   │   └── partialspoof # [ASRU2025 Foucs]
│   │       ├── README 	 # which version refer to which model.
│   │       ├── fusion	 # fusing N models' results. 
│ 	│  		  │      └──v03_12 # fusion of v03 and v12.
│   │       ├── v03_resnet18
│   │       ├── v12_ssl_res1d
│   │       ├── v15_ssl_mhfa
│   │       └── x12_ssl_res1d
│   ├── localization
│   │   ├── llamapartialspoof
│   │   ├── partialedit
│   │   └── partialspoof
│   ├── diarization #[Future]
│   ├── sasv #[Future]
│   └── source_tracing #[Future]
├── LICENSE
├── README.md
├── requirements.txt
├── tools #[same as in wespeaker]
│   ├── combine_data.sh
│   ├── copy_data_dir.sh
│   ├── extract_embedding.sh
│   ├── filter_scp.pl
│   ├── fix_data_dir.sh
│   ├── generate_calibration_trial.py
│   ├── make_feat_list.py
│   ├── make_lmdb.py
│   ├── make_raw_list.py
│   ├── make_shard_list.py
│   ├── parse_options.sh
│   ├── spk2utt_to_utt2spk.pl
│   ├── subset_data_dir.sh
│   ├── utt2spk_to_spk2utt.pl
│   ├── vector_mean.py
│   ├── wav2dur.py
│   └── wav_to_duration.sh
└── wedefense #[main modules]
    ├── fusion #TODO? folder to implement fusion [Johan]
		├── calibration # TODO? folder to save calibration related recipts. [Johan]  
    ├── bin
    │   ├── average_model.py
    │   ├── extract.py
    │   └── train.py
    ├── dataset #collate function, augmentation
    │   ├── customize_collate_fn.py #copy from Xin's code
    │   ├── customize_sampler.py	  #copy from Xin's code
    │   ├── dataset.py
    │   ├── dataset_utils.py
    │   ├── __init__.py
    │   ├── lmdb_data.py
    │   ├── processor.py
    │   ├── rawboost.py: copy from Hemlata\'s code
    │   └── rawboost_util.py: copy from Hemlata\'s code
    ├── diarization #[Future]
    ├── frontend #[Junyi's SSL?]
    │   ├── __init__.py
    │   ├── s3prl.py
    │   └── whisper_encoder.py
    ├── __init__.py
    ├── localization
    ├── metrics
    │   ├── significiation_testing #[put to here?]
    │   ├── confidence_intervals #[put to here?]
    │   ├── localization
    │   └── detection
    │       └── asvspoof5
    │           ├── a_dcf.py
    │           ├── calculate_metrics_full.py
    │           ├── calculate_metrics.py
    │           ├── calculate_modules.py
    │           ├── evaluation_full.py
    │           ├── evaluation.py
    │           ├── README.md
    │           ├── util.py
    │           └── util_table.py
    ├── models #models are added here. (those from wespeaker will be move to pip installed wespeaker)
    │   ├── campplus.py #wedefense will not consider large-margin finetuning for now?
    │   ├── __init__.py
    │   ├── pooling_layers.py
    │   ├── projections.py
    │   ├── resnet.py
    │   └── speaker_model.py
    ├── utils
    │   ├── checkpoint.py
    │   ├── executor_deprecated.py
    │   ├── executor.py
    │   ├── file_utils.py
    │   ├── schedulers.py
    │   ├── score_metrics.py
    │   └── utils.py
    └── xai # explainable AI

```



## TODO

Till 0511:

[Lin]: 

1. add varied-length input, 

[Junyi]:

1. SSL supporting

[Lin, Shuai] move wespeaker part to pip install wespeaker





Reference:
1. Mainly adopated from [wespeaker](https://github.com/wenet-e2e/wespeaker)
