<div align="center">

## ⚙️ UniPCB: Unifying Generation and Detection for PCB Defect Inspection

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2605.04635)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange.svg)](https://pytorch.org/)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=House-yuyu.UniPCB)

</div>

---

This is the official PyTorch codes for the paper:

>**UniPCB: Unifying Generation and Detection for PCB Defect Inspection**<br>  [Huan Zhang<sup>1</sup>](https://scholar.google.com.hk/citations?user=bJjd_kMAAAAJ&hl=zh-CN), [Lianghong Tan<sup>1</sup>](), [Yichu Xu<sup>2</sup>](), [Jiangzhong Cao<sup>1📧</sup>](), [Huanqi Wu<sup>1</sup>](), [Linwei Zhu<sup>3</sup>](), [Xu Zhang<sup>2📧</sup>](https://scholar.google.com.hk/citations?hl=zh-CN&view_op=list_works&gmla=AIqSsVv0Hi6v-HFtCPhkMQfNslqeoLmOAJsutMfgrvgOo7bff33oYvwukfAev9Q6ISPykrPHVCJfU2CU-q_B0cFlX1u8IHF0MRzgnxgsfFMMT7cB6HH3KDcaugHfTTDbx4Y&user=xDDy-DwAAAAJ)<br>
> <sup>1</sup>Guangdong University of Technology, <sup>2</sup>Wuhan University, <sup>3</sup>Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences
> <sup>📧</sup> Corresponding author

![teaser_img](gen_model.png)
![teaser_img](det_model.png)

## 📖 Overview

**UniPCB** is a **generation-assisted PCB defect inspection framework** that unifies multimodal controllable defect synthesis with feature-enhanced detection, jointly alleviating data scarcity and insufficient representation within a single model, combining conditional image generation (`PCB_control`) and defect detection (`PCB_detect`).

## ✨ Highlights

- **UniPCB**: A generation-assisted PCB defect inspection pipeline to systematically enhance inspection performance.
- **Multi-modal Controlled Defect Synthesis**: A latent diffusion architecture with multi-scale embedding and conditional modulation to improve sample fidelity and diversity.
- **Feature-Enhanced Defect Detection**: Effectively capture both global contextual dependencies and fine-grained local textures.

## PCB_control

`PCB_control` implements the controllable PCB defect generation module. It supports training with multimodal conditions and can be adapted to customized PCB defect datasets.

### Training

#### 1. Download Pretrained Weights

Download the pretrained Stable Diffusion weights `v1-5-pruned.ckpt` and place them under:

```bash
PCB_control/ckpt/
```

#### 2. Prepare Initialization Weights

Run the following command to initialize the training weights:

```bash
cd PCB_control

python utils/prepare_weights.py init_local \
    ckpt/v1-5-pruned.ckpt \
    configs/local_v15.yaml \
    ckpt/init_local.ckpt
```

The arguments are organized as follows:

```text
mode | pretrained SD weights | config file | output path
```

#### 3. Prepare Training Data

Please organize the training data under the `PCB_control/data/` directory as follows:

```text
PCB_control/
└── data/
    ├── anno.txt
    ├── images/
    │   ├── xxx.png
    │   ├── yyy.png
    │   └── ...
    └── conditions/
        ├── condition-1/
        │   ├── xxx.png
        │   ├── yyy.png
        │   └── ...
        ├── condition-2/
        │   ├── xxx.png
        │   ├── yyy.png
        │   └── ...
        └── ...
```

Specifically:

- Place the original PCB images in:

```bash
PCB_control/data/images/
```

- Place the extracted condition maps in:

```bash
PCB_control/data/conditions/condition-N/
```

- Prepare the annotation file:

```bash
PCB_control/data/anno.txt
```

Each line in `anno.txt` should contain two parts:

```text
file_id annotation
```

Please make sure that the file IDs are consistent across:

```text
PCB_control/data/anno.txt
PCB_control/data/images/
PCB_control/data/conditions/condition-N/
```

#### 4. Start Training

After preparing the data and initialization weights, run:

```bash
cd PCB_control

python src/train/train.py
```

Please note that the **local adapter** and **global adapter** need to be trained separately. You can customize the training settings in:

```text
PCB_control/src/train/train.py
PCB_control/configs/
```

---

## PCB_detect

`PCB_detect` implements the defect detection module. The customized modules and model variants are built upon the [Ultralytics YOLO](https://docs.ultralytics.com) framework.

All training, validation, and inference workflows follow the official Ultralytics usage.

### Usage Instructions

#### 1. Model Configuration

The customized model configuration files are located in:

```bash
PCB_detect/ultralytics/cfg/models/rt-detr/
```

During training, specify the corresponding YAML configuration file as the model path.

#### 2. Training

Please refer to the official Ultralytics training guide:

```text
https://docs.ultralytics.com/modes/train/
```

Example:

```bash
yolo task=detect \
    mode=train \
    model="path/to/rtdetr-IRSA-CAMF.yaml" \
    data="path/to/dataset.yaml" \
    device=0,1 \
    pretrained=False \
    imgsz=512
```

#### 3. Validation

You can validate a trained model using:

```bash
yolo task=detect \
    mode=val \
    model="path/to/rtdetr_irsa_camf.pt" \
    data="path/to/dataset.yaml" \
    device=0 \
    imgsz=512
```

#### 4. Inference

Please refer to the official Ultralytics inference guide:

```text
https://docs.ultralytics.com/modes/predict/
```

Example:

```bash
yolo predict \
    imgsz=512 \
    model="path/to/rtdetr_irsa_camf.pt" \
    source="path/to/images/" \
    device=0
```
