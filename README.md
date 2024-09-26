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
Please download from [SAM](https://github.com/facebookresearch/segment-anything) and extract them under "models/Pretrained_model".

```
models
├── Pretrained_model
    ├── sam_vit_b_01ec64.pth
```

## Dataset

## Fine tuning on SAM
### Binary segmentation (ISBI2012)
If we prepared ISBI2012 dataset, we can directly run the following code to train the model with single GPU.
```
python3 train.py --gpu 0 --dataset 'ISBI2012' --out result_sam --modelname 'SAM' --batchsize 8
```
If we want to utilize multi GPUs, we can directly run the following code.
```
CUDA_VISIBLE_DEVICES=0,1 python3 train.py --dataset 'ISBI2012' --out result_sam --modelname 'SAM' --batchsize 8 --multi
```

### Multi-class segmentation (ssTEM)
If we prepared ssTEM dataset, we can directly run the following code to train the model with single GPU.
```
python3 train.py --gpu 0 --dataset 'ssTEM' --out result_sam --modelname 'SAM' --batchsize 8 --num_classes=5 --multimask_output=True
```

## Fine tuning on SAM with Anything
We can try to use variour adaptation methods.
Please run the following code to train the improved SAM.

### Fine tuning with LoRA [[paper]](https://arxiv.org/abs/2106.09685)
```
python3 train.py --gpu 0 --dataset 'ISBI2012' --modelname 'SAM_LoRA' 
```

### Fine tuning with ConvLoRA [[paper]](https://arxiv.org/abs/2401.17868)
```
python3 train.py --gpu 0 --dataset 'ISBI2012' --modelname 'SAM_ConvLoRA'
```

### Fine tuning with AdaptFormer [[paper]](https://arxiv.org/abs/2205.13535)
```
python3 train.py --gpu 0 --dataset 'ISBI2012' --modelname 'SAM_AdaptFormer'
```

### Fine tuning with SAMUS [[paper]](https://arxiv.org/abs/2309.06824)
```
python3 train.py --gpu 0 --dataset 'ISBI2012' --modelname 'SAMUS'
```



## Citation
```
@article{kato2024generalized,
  title={Generalized SAM: Efficient Fine-Tuning of SAM for Variable Input Image Sizes},
  author={Kato, Sota and Mitsuoka, Hinako and Hotta, Kazuhiro},
  journal={arXiv preprint arXiv:2408.12406},
  year={2024}
}
```
