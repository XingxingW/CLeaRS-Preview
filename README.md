# CLeaRS-Preview

Official repository for our paper: 

> **[Continual Vision-Language Learning for Remote Sensing: Benchmarking and Analysis]**  
> Xingxing Weng, Ruifeng Ni, Chao Pang, XiangYu Hao, Yishan Wang, Xiaokang Zhang, Wei Xu, Gui-Song Xia

---

## Overview

CLeaRS is a comprehensive benchmark designed to investigate continual learning in remote sensing vision-language models. It comprises 10 subsets with over 207k image-text pairs, covering diverse interpretation tasks, sensing modalities, and application scenarios. Based on CLeaRS, we define three evaluation protocols, namely long-horizon, modality-incremental, and task-incremental settings, to systematically assess model adaptability under evolving data streams.

![teaser](CLeaRS-example.png)

## Coming Soon

-[ ] Benchmark dataset

-[ ] Training & inference code

-[ ] Model checkpoints

## Acknowledgement

Our CLeaRS dataset is built based on AID, VRSBench, SARDet-100K, SARLANG-1M, DroneVehicle, FireRisk, and RescueNet-VQA datasets.
We are thankful to Qwen2.5-VL, MiniGPT-v2, LLaVA-1.5, GeoChat, and VHM for releasing their models and code as open-source contributions.
