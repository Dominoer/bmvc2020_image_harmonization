# Image Harmonization with Attention-based Deep Feature Modulation

This code provides a pytorch implementation of "[Image Harmonization with Attention-based Deep Feature Modulation](https://www.bmvc2020-conference.com/assets/papers/0121.pdf)".

[Project Page](https://bmvc2020-conference.com/conference/papers/paper_0121.html)

## Requirements
- Python 3.7
- [PyTorch](https://pytorch.org/) tested on 1.4.0
- json
- PIL
- skimage
- tqdm
- numpy

## Dataset
Please download the iHarmony4 dataset from [this link](https://github.com/bcmi/Image_Harmonization_Datasets).

## Colab demo (coming soon)

## Training
```
bash train.sh
```
## Testing
```
bash test.sh
```
## Citation
If you find the code useful in your research, please consider citing our paper:
```
@InProceedings{Hao2020bmcv,
author       = "Guoqing, Hao and Satoshi, Iizuka and Kazuhiro, Fukui",
title        = "Image Harmonization with Attention-based Deep Feature Modulation",
booktitle    = "The British Machine Vision Conference (BMCV)",
year         = "2020",
}
```