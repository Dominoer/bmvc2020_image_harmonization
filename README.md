# Image Harmonization with Attention-based Deep Feature Modulation

This code provides a pytorch implementation of "[Image Harmonization with Attention-based Deep Feature Modulation](https://www.bmvc2020-conference.com/assets/papers/0121.pdf)".

[Project Page](https://dominoer.github.io/bmvc2020_image_harmonization/)

## Requirements
- Python 3.7
- [PyTorch](https://pytorch.org/) tested on 1.4.0
- PIL
- skimage
- tqdm
- numpy

## Dataset
Please download the iHarmony4 dataset from [this link](https://github.com/bcmi/Image_Harmonization_Datasets).

We resized all training images of HAdobe5k subset with a max size of 1024. Original images of HAdobe5k are quite large, which will slow down the data loading time. In order to speed up this process, we resize all training images of HAdobe5k dataset. 

To resize the training images of HAdobe5k, specify the paths in the following script and run the script. 
```
python resize.py
```
## Training
Specify the paths in the train.sh script before runing the following script.
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