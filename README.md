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

**Note:** This is **NOT** the same as the repo version. It's for reference and discussion only, intended to guide future restructuring. Final decisions will need to be confirmed with co-authors. (try to decide on May 11)

```shell
.
├── egs     # example folders to include supported tasks/databases.
│   ├── detection
│   │   ├── asvspoof5 
│   │   ├── llamapartialspoof
│   │   ├── partialedit
│   │   └── partialspoof				# [ASRU2025 Foucs]
│   │       ├── README					# Describe which version refer to which model.
│   │       ├── v03_resnet18
│   │       ├── v12_ssl_res1d
│   │       ├── v15_ssl_mhfa
│   │       ├── x12_ssl_res1d
│   │       └── fus_v03_12					# example for fusion, fusion of v03 and v12.
│   └── localization
│       ├── llamapartialspoof
│       ├── partialedit
│       └── partialspoof
├── LICENSE
├── README.md
├── requirements.txt
├── tools  #[same as in wespeaker]
└── wedefense	 #[main modules]
    ├── bin
    │   ├── average_model.py
    │   ├── extract.py
    │   └── train.py
    ├── calibration						# folder to implement calibration [Johan]
    ├── dataset
    │   ├── acoustic_feature
    │   ├── augmentation
    │   │   ├── rawboost.py				#copy from Hemlata's code
    │   │   └── rawboost_util.py	#copy from Hemlata's code
    │   ├── customize_collate_fn.py  #copy from Xin's code
    │   ├── customize_sampler.py 		 #copy from Xin's code
    │   ├── dataset.py
    │   ├── dataset_utils.py
    │   ├── __init__.py
    │   ├── lmdb_data.py
    │   ├── processor.py
    │   └── shuffle_random_tools.py
    ├── frontend
    │   ├── __init__.py
    │   ├── s3prl.py
    │   └── whisper_encoder.py
    ├── fusion 				# folder to implement fusion [Johan]
    ├── __init__.py
    ├── metrics
    │   ├── confidence_intervals
    │   ├── detection
    │   │   ├── a_dcf.py
    │   │   ├── calculate_metrics_full.py
    │   │   ├── calculate_metrics.py
    │   │   ├── calculate_modules.py
    │   │   ├── evaluation_full.py
    │   │   ├── evaluation.py
    │   │   ├── README.md
    │   │   ├── util.py
    │   │   └── util_table.py
    │   └── localization
    ├── models  #models are added here. (those from wespeaker will be move to pip installed wespeaker in future)
    │   ├── campplus.py  #wedefense will not consider large-margin finetuning for now.
    │   ├── __init__.py
    │   ├── loss # For different loss functions, need to clean from projections.py
    │   ├── pooling_layers.py
    │   ├── projections.py
    │   ├── resnet.py
    │   └── speaker_model.py
    └── utils
        ├── checkpoint.py
        ├── executor_deprecated.py
        ├── executor.py
        ├── file_utils.py
        ├── __init__.py
        ├── schedulers.py
        ├── score_metrics.py
        └── utils.py

```

## Contribute (For coauthors)
0. Adopted rule from wespeaker (https://wenet.org.cn/wespeaker/contribute.html)

1. Git clone to your local machine [may not need to fork right now?]
2. Create a new branch for your changes:
    * `feature/xxx` for new features
    * `database_name/model_name/xxx` changes for specific model under a database
    * `bugfix/xxx` for bug fixes
    * etc.
3. Make your changes and ensure the code runs successfully on your local environment.
4. Commit your changes with a meaningful message:
   `git commit -am '[type] Brief description of your update` 
5. Push your changes to your remote branch
6. Create a Pull Request (PR) on GitHub:
    Please include a detailed description of your changes and the purpose of the update.
7. We recommend following the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) 
    for code formatting and consistency. But don't worry too much given the tight timeline.
    we will also do this again before release.


## TODO

For the main structure of wedefense: (need to be implemented ASAP)

Lin: localization
Junyi: add an example for SSL 
Johan: calibration, fusion
Lin, Shuai: move wespeaker part to pip install wespeaker



Other TODOs please refer to our overleaf.



## Reference:

1. Mainly adopated from [wespeaker](https://github.com/wenet-e2e/wespeaker)
