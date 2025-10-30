# WeDefense Support Status

✅ Supported

⏳ Planning



## Databases

TODO: check type

| Name                | Release | Links                                                        | Type            | Tasks          | Support |
| ------------------- | ------- | ------------------------------------------------------------ | --------------- | -------------- | ------- |
| <u>**ASVspoof**</u> |         |                                                              |                 | det            |         |
| ASVspoof2019        | 2019    |                                                              | TTS/VC          | det            | ⏳       |
| ASVspoof2021        | 2021    |                                                              | TTS/VC          | det            |         |
| ASVspoof 5 (2024)   | 2024    | paper, database                                              | TTS/VC          | det (TD: sasv) | ✅       |
| ADD 2022            | 2022    |                                                              | TTS/Real        | det, loc       |         |
| ADD 2023            | 2023    |                                                              | TTS/Real        | det, loc       |         |
| In-the-wild (ITW)   |         |                                                              | Unk.            | det            | ⏳       |
| ODSS                | 2023    | [paper](https://ieeexplore.ieee.org/document/10374863)       | TTS/VC          | det            | ⏳       |
| ShiftySpeech        | 2025    |                                                              | Vocoder         | det            |         |
| MLAAD               | 2024    | [paper](https://arxiv.org/abs/2401.09512)                    | TTS/VC          | det            | ⏳       |
| SpoofCeleb          | 2025    | [paper](https://arxiv.org/abs/2409.17285)                    | TTS/VC          | det, sasv      | ⏳       |
| ReplayDF            |         |                                                              | Reply           | det            |         |
| DeePen              | 2025    | [paper](https://arxiv.org/abs/2502.20427) \| [github](https://github.com/Fraunhofer-AISEC/DeePen) | TTS/VC          | det            |         |
| FoR                 |         | [paper](https://ieeexplore.ieee.org/document/8906599)        | TTS/VC          | det            |         |
| DFADD               |         | [paper](https://arxiv.org/abs/2409.08731)                    | TTS (diffusion) | det            |         |
| AI4T                |         | [paper](https://arxiv.org/abs/2506.09606)                    | TTS/VC          | det            |         |
| Deepfake-Eval-2024  |         | [paper](https://arxiv.org/abs/2503.02857)                    | TTS/VC          | det            |         |
| SpeechFake          |         | [paper](https://openreview.net/forum?id=GpUO6qYNQG)          | TTS/VC          | det            |         |
|                     |         |                                                              |                 |                |         |
|                     |         |                                                              |                 |                |         |
| AV-Deepfake1M       | 2024    | [data](https://huggingface.co/datasets/ControlNet/AV-Deepfake1M) | TTS (av)        | det, loc       | ⏳       |
| AV-Deepfake1M++     | 2025    | [data](https://huggingface.co/datasets/ControlNet/AV-Deepfake1M-PlusPlus) | TTS (av)        | det, loc       | ⏳       |
| HAD                 | 2021    | [data](https://zenodo.org/records/10377492)                  |                 |                | ⏳       |
| Psynd               | 2022    | [data](https://scholarbank.nus.edu.sg/handle/10635/227398)   |                 |                |         |
| PartialSpoof        | 2021    | [data](https://zenodo.org/records/5766198)                   | TTS/VC          | det, loc,      | ✅       |
| LAV-DF              | 2022    | [data](https://huggingface.co/datasets/ControlNet/LAV-DF)    | TTS (av)        |                |         |
| LlamaPartialSpoof   | 2025    | [data](https://huggingface.co/datasets/HaoY0001/LlamaPartialSpoof) | TTS             | Det, loc       | ⏳       |
| PartialEdit         | 2025    | [paper]()                                                    | TTS             |                | ⏳       |
|                     |         |                                                              |                 |                |         |
|                     |         |                                                              |                 |                |         |



## Augmentation

| Name          | Date | Links | Comment | **Support** |
| ------------- | ---- | ----- | ------- | ----------- |
| speed perturb |      |       |         |             |
| codec         |      |       |         |             |
| Raw boost     |      |       |         |             |
| musan + rirs  |      |       |         |             |
|               |      |       |         |             |
|               |      |       |         |             |
|               |      |       |         |             |
|               |      |       |         |             |



## Detection

| Name                        | Date | Links                                                        | Comment | **Support** |
| --------------------------- | ---- | ------------------------------------------------------------ | ------- | ----------- |
| <u>**Normal NN**</u>        |      |                                                              |         |             |
| LCNN-LSTM                   |      |                                                              |         | ✅           |
| ResNet                      |      |                                                              |         | ✅           |
| AASIST                      |      |                                                              |         | ⏳           |
| <u>**SSL-based models**</u> |      |                                                              |         |             |
| SSL-gmlp                    | 2022 | [paper](https://arxiv.org/abs/2105.08050) \| [paper2](https://ieeexplore.ieee.org/document/10003971) \| [code](https://github.com/nii-yamagishilab/PartialSpoof/tree/main) |         | ✅           |
| SSL-AASIST                  | 2022 | [paper](https://arxiv.org/abs/2202.12233) \|[code](https://github.com/TakHemlata/SSL_Anti-spoofing) |         | ✅           |
| SSL_MHFA                    | 2022 | [Paper](https://ieeexplore.ieee.org/document/10022775)       |         | ✅           |
| SSL-res1d                   | 2024 | [paper](https://www.isca-archive.org/interspeech_2024/liu24m_interspeech.pdf) |         | ✅           |
| SSL-SLS                     | 2024 | [paper](https://openreview.net/pdf?id=acJMIXJg2u) \| [code](https://github.com/QiShanZhang/SLSforASVspoof-2021-DF) |         | ✅           |
|                             |      |                                                              |         |             |
|                             |      |                                                              |         |             |
|                             |      |                                                              |         |             |
|                             |      |                                                              |         |             |
|                             |      |                                                              |         |             |
|                             |      |                                                              |         |             |
| <u>**LLM-based models**</u> |      |                                                              |         |             |
| ALLM4ADD                    | 2025 | [paper](https://arxiv.org/abs/2505.11079)                    |         | ⏳           |
|                             |      |                                                              |         |             |



## Localization

TODO: check type

| Name                            | Date | Links                                                        | Comment                     | Support |
| ------------------------------- | ---- | ------------------------------------------------------------ | --------------------------- | ------- |
| **<u>Uniform segmentation</u>** |      |                                                              |                             |         |
| SSL-MHFA                        |      |                                                              |                             | ✅       |
| SSL-SLS                         |      |                                                              |                             | ✅       |
|                                 |      |                                                              |                             |         |
| **<u>Boundary-aware</u>**       |      |                                                              |                             |         |
| SSL-BAM                         | 2025 | [paper](https://arxiv.org/abs/2407.21611) \| [code](https://github.com/media-sec-lab/BAM) |                             | ✅       |
| CFPRF                           | 2024 | [paper](https://arxiv.org/abs/2407.16554) \| [code](https://github.com/ItzJuny/CFPRF) |                             |         |
|                                 |      |                                                              |                             |         |
| PET                             | 2024 | [paper](https://ieeexplore.ieee.org/document/10889913/)      |                             |         |
| AGO                             | 2025 | [paper](https://ieeexplore.ieee.org/document/10890470)       |                             |         |
| GNCL                            | 2025 | [paper](https://ieeexplore.ieee.org/document/10888281)       |                             |         |
| UMMAformer                      | 2023 | [paper](https://arxiv.org/abs/2308.14395) \| [code](https://github.com/ymhzyj/UMMAFormer) |                             | ⏳       |
| W-TDL                           | 2024 | [paper](https://dl.acm.org/doi/10.1145/3689092.3689410)      |                             |         |
| BA-TFD                          | 2022 | [paper](https://arxiv.org/abs/2204.06228) \|[code](https://github.com/ControlNet/LAV-DF) | baseline for AV-Deepfake-1M | ⏳       |
| BA-TFD+                         | 2023 | [paper](https://arxiv.org/abs/2305.01979) \|[code](https://github.com/ControlNet/LAV-DF) | baseline for AV-Deepfake-1M | ⏳       |
| VIGO                            | 2024 | [paper](https://dl.acm.org/doi/10.1145/3664647.3688983)      |                             |         |
|                                 |      |                                                              |                             |         |
|                                 |      |                                                              |                             |         |
|                                 |      |                                                              |                             |         |
|                                 |      |                                                              |                             |         |
|                                 |      |                                                              |                             |         |
|                                 |      |                                                              |                             |         |
|                                 |      |                                                              |                             |         |
|                                 |      |                                                              |                             |         |
|                                 |      |                                                              |                             |         |
|                                 |      |                                                              |                             |         |
|                                 |      |                                                              |                             |         |
|                                 |      |                                                              |                             |         |
|                                 |      |                                                              |                             |         |



