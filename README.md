# 3DFP_FCGAN

 Implementation of paper "3DFP-FCGAN : Face Completion Generative Adversarial Network with 3D Facial Prior" [2021JVCIR].
 
## Configuration Environment

python 3.7.0 + tensorflow 1.13.0 + neuralgym 0.0.1

## Preparation

### 1. Dataset

For CelebA: download the cropped face images.
For CelebA HQ: dowmload the face images and then run `cropped_image.py` to generate cropped face images.

### 2. pretrained model

Download the pretrained model from [https://pan.baidu.com/s/1Ib1KQdcNLg9u1hUX1_e9CA](https://pan.baidu.com/s/1Ib1KQdcNLg9u1hUX1_e9CA) [Passeword: vlcv]

### For evaluation

Run `test.py`.

## Acknowledgement

Thanks to previous open-sourced repo:

[Generative image inpainting](https://github.com/JiahuiYu/generative_inpainting)
