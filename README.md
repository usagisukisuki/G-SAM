# Generalized SAM
This repo is the official implementation for *Generalized SAM: Efficient Fine-Tuning of SAM for Variable Input Image Sizes*.
This research was accepted by ECCV2024 Workshop "Computational Aspects of Deep Learning (CADL)"


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
