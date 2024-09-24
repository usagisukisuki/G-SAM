# Generalized SAM: Efficient Fine-Tuning of SAM for Variable Input Image Sizes (ECCV2024 Workshop)
[![arXiv](https://img.shields.io/badge/arXiv-2408.12406-b31b1b.svg)](https://arxiv.org/pdf/2408.12406)


This repo is the official implementation for *Generalized SAM* accepted by ECCV2024 Workshop [*Computational Aspects of Deep Learning (CADL)*](https://sites.google.com/nvidia.com/cadl2024).

## Highlights
<div align="center">
  <img src="figs/img1.png" width="60%"> <img src="figs/img2.png" width="30%">
</div>


- **Training using random cropping**: Our Generalized SAM (GSAM) can cope with variable input image sizes, allowing random cropping to be used the first time during fine-tuning for SAM.
- **Multi-scalce AdaptFormer**: GSAM can use multi-scale features during fine-tuning for SAM.
- **Low computational cost of training**: compared to the conventional SAM fine-tuning methods, GSAM can significantly reduce the computational training cost and GPU memories.


## Installation
Following [Segment Anything](https://github.com/facebookresearch/segment-anything), `python=3.8.16`, `pytorch=1.8.0`, and `torchvision=0.9.0` are used in GSAM.
1. Clone this repository.
   ```
   git clone https://github.com/usagisukisuki/G-SAM.git
   cd G-SAM
   ```
2. Install Pytorch and TorchVision. (you can follow the instructions here)
3. Install other dependencies.
   ```
   pip install -r requirements.txt
   ```

## Checkpoints
We use checkpoint of SAM in [vit_b](https://github.com/facebookresearch/segment-anything) version.
Additionally, we also use checkpoint of MobileSAM.
Please download from [SAM](https://github.com/facebookresearch/segment-anything) and [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), and extract them under "models/Pretrained_model".

```
models
├── Pretrained_model
    ├── sam_vit_b_01ec64.pth
    ├── mobile_sam.pt
```

## Dataset

## Fine tuning on SAM



## Citation
```
@article{kato2024generalized,
  title={Generalized SAM: Efficient Fine-Tuning of SAM for Variable Input Image Sizes},
  author={Kato, Sota and Mitsuoka, Hinako and Hotta, Kazuhiro},
  journal={arXiv preprint arXiv:2408.12406},
  year={2024}
}
```
