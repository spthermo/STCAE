## A Deep Learning Approach To Object Affordance Segmentation
This repo contains code and data samples from our paper "A Deep Learning Approach To Object Affordance Segmentation" (to appear in [ICASSP 2020](https://2020.ieeeicassp.org/)). 

## Model Overview

![concept](./sor3d-aff_samples/concept.png)

## Prerequisites
The following are the minimum requirements to replicate the paper experiments:
- Python 3.7.2
- PyTorch 1.0.1
- CUDA 9.1
- Visdom (follow the steps [here](https://github.com/facebookresearch/visdom))

## SOR3D-AFF samples

### RGB - original resolution (1920x1080)
![rgb_full](./sor3d-aff_samples/rgb_full.png)

### RGB - aligned with depth maps (512x424)
![rgb_aligned](./sor3d-aff_samples/rgb_aligned.png)

### 3D optical flow - after preprocessing (300x300)
![3Dflow](./sor3d-aff_samples/3Dflow.png)

### Segmentation masks - Last frame only (512x424)
![seg_mask](./sor3d-aff_samples/seg_mask.png)

## Train
**ToDo**
## Test
**ToDo**

# Citation
If you use any code or model from this repo, please cite the following:
```
@inproceedings{thermos2020affordance,
  author       = "Spyridon Thermos and Petros Daras and Gerasimos Potamianos",
  title        = "A Deep Learning Approach To Object Affordance Segmentation",
  booktitle    = "Proc. International Conference on Acoustics Speech and Signal Processing (ICASSP)",
  year         = "2020"
}
```

# License
Our code is released under MIT License (see LICENSE file for details)
