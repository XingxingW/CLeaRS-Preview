# CLeaRS-Preview

Official repository for our paper: 

> **[Continual Vision-Language Learning for Remote Sensing: Benchmarking and Analysis]**  
> Xingxing Weng, Ruifeng Ni, Chao Pang, XiangYu Hao, Yishan Wang, Xiaokang Zhang, Wei Xu, Gui-Song Xia

---

## Overview

CLeaRS is a comprehensive benchmark designed to investigate continual learning in remote sensing vision-language models. It comprises 10 subsets with over 207k image-text pairs, covering diverse interpretation tasks, sensing modalities, and application scenarios. Based on CLeaRS, we define three evaluation protocols, namely long-horizon, modality-incremental, and task-incremental settings, to systematically assess model adaptability under evolving data streams.

![teaser](CLeaRS-example.png)

[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/cdllwhy/clears)

## Key Findings
* RS VLMs suffer from catastrophic forgetting, influenced by transitions in tasks, instructions, and sensing modalities.
* Existing continual learning methods struggle to balance stability and plasticity, and fail to deliver consistent gains over sequential fine-tuning across different settings and learning orders.
* CLeaRS highlights the need for continual learning methods tailored for RS VLMs.

## Datasets

CLeaRS is built upon the following publicly released remote sensing datasets. Please download each subset from its official source and follow the corresponding license.
The CLeaRS annotations are hosted on Hugging Face: **[cdllwhy/clears](https://huggingface.co/datasets/cdllwhy/clears)**.

| Dataset | Source |
| --- | --- |
| AID | https://captain-whu.github.io/AID/ |
| VRSBench | https://huggingface.co/datasets/xiang709/VRSBench |
| SARDet-100K | https://github.com/zcablii/SARDet_100K |
| DroneVehicle | https://github.com/VisDrone/DroneVehicle |
| FireRisk | https://github.com/CharmonyShen/FireRisk |
| RescueNet-VQA | https://github.com/BinaLab/RescueNet-VQA |

#### Example Data Organization

All images are placed under `images/`, with one sub-folder per source dataset, and all annotation files are grouped under `annotations/`, split into `train/` and `test/`:

```
CLeaRS/
├── images/
│   ├── AID/                    # https://captain-whu.github.io/AID/
│   ├── VRSBench/               # https://huggingface.co/datasets/xiang709/VRSBench
│   ├── SARDet_100K/            # https://github.com/zcablii/SARDet_100K
│   ├── DroneVehicle/           # https://github.com/VisDrone/DroneVehicle
│   ├── FireRisk/               # https://github.com/CharmonyShen/FireRisk
│   └── RescueNet-VQA/          # https://github.com/BinaLab/RescueNet-VQA
├── annotations/
    ├── train/
    └── test/
```

## Coming Soon

* Benchmark dataset
* Training & inference code
* Model checkpoints

**Note:** The benchmark dataset is currently being prepared for release. If you need access to the dataset before its public release, please contact us via email (xingxingw@whu.edu.cn) for a private copy.

## Acknowledgement

Our CLeaRS dataset is built based on AID, VRSBench, SARDet-100K, SARLANG-1M, DroneVehicle, FireRisk, and RescueNet-VQA datasets.

We are thankful to Qwen2.5-VL, MiniGPT-v2, LLaVA-1.5, GeoChat, and VHM for releasing their models and code as open-source contributions.