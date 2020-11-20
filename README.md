# [Unmixing Convolutional Features for Crisp Edge Detection](http://arxiv.org/abs/2011.09808)



## Requirements

pytorch >= 1.0

torchvision

opencv-python

tqdm



## Training  & Testing

### Data preparation

- Download the test data for  [BSDS500 dataset and the NYUDv2 dataset](https://drive.google.com/file/d/1AqD3q-xeTD_HNh4wzDVvU3rXTAF9qRu_/view?usp=sharing).
- The training data will be updated soon.
- Place the images to "./data/.
- The default testing data is BSDS500. For getting the inference results of NYUDv2 dataset, one can change the 8th line in configs/__init__.py as follows.

```python
class Config(object):
    def __init__(self):
        self.data = "nyud"
```

The structure of the data folder should be

```shell
./data
   bsds/test/*
   bsds/train/*
   bsds/test.lst
   bsds/train.lst
   ------------------------
   nyud/test/*
   nyud/train/*
   nyud/test.lst
   nyud/train.lst
```

#### 

### Pretrained Models

- Download the pretrained model and unzip the model to **"./pretrained/"**
- [Pretrained model for BSDS500](https://drive.google.com/file/d/1xWYCKjdJTzSREYC9DHbUfZLViOf2CaME/view?usp=sharing)
- [Pretrained model for NYUDv2](https://drive.google.com/file/d/11DuMk38ZcPnnBuyP_ukpGODHJQkI5p-7/view?usp=sharing)



### Training

```shell
python main.py --mode train
```

The training data will be updated soon.



### Testing

```shell
python main.py --mode test
```

The output results will be saved to ./output/$dataset_name/single_scale_test/



## Quantitative Comparison



### BSDS500

|          Method          |      ODS      |      OIS      |
| :----------------------: | :-----------: | :-----------: |
| HED(official/retrained)  | 0.790 / 0.793 | 0.808 / 0.811 |
| RCF(official/retrained)  | 0.798 / 0.799 | 0.815 / 0.815 |
| BDCN(official/retrained) | 0.806 / 0.807 | 0.826 / 0.822 |
|         CATS-HED         |     0.800     |     0.816     |
|         CATS-RCF         |     0.805     |     0.822     |
|        CATS-BDCN         |     0.812     |     0.828     |



### NYUDv2

|          Method          |      ODS      |      OIS       |
| :----------------------: | :-----------: | :------------: |
| HED(official/retrained)  | 0.720 / 0.722 | 0..734 / 0.737 |
| RCF(official/retrained)  | 0.743 / 0.745 | 0.757 / 0.759  |
| BDCN(official/retrained) | 0.748 / 0.748 | 0.763 / 0.762  |
|         CATS-HED         |     0.732     |     0.746      |
|         CATS-RCF         |     0.752     |     0.765      |
|        CATS-BDCN         |     0.752     |     0.765      |




## Visualization Results
There are some visualized results in './examples'.

More results can be downloaded from links below.

- The visualization results of 
- The visualization results of [NYUDv2](https://drive.google.com/file/d/15lKMRWPKFxEn06Lnk1rl0iBaId4EfJSc/view?usp=sharing)



### Acknowledgment

We acknowledge the effort from the authors of HED, RCF and BDCN on edge detection. Their researches laid the foundation for this work. We thank [meteorshowers](https://github.com/meteorshowers/RCF-pytorch) as this code is based on  the reproduced RCF of pytorch version by [meteorshowers](https://github.com/meteorshowers/RCF-pytorch).

```
@article{xie2017hed,
author = {Xie, Saining and Tu, Zhuowen},
journal = {International Journal of Computer Vision},
number = {1},
pages = {3--18},
title = {Holistically-Nested Edge Detection},
volume = {125},
year = {2017}
}

@article{liu2019richer,
author = {Liu, Yun and Cheng, Ming-Ming and Hu, Xiaowei and Bian, Jia-Wang and Zhang, Le and Bai, Xiang and Tang, Jinhui},
journal = {IEEE Trans. Pattern Anal. Mach. Intell.},
number = {8},
pages = {1939--1946},
publisher = {IEEE},
title = {Richer Convolutional Features for Edge Detection},
volume = {41},
year = {2019}
}

@inproceedings{he2019bi-directional,
author = {He, Jianzhong and Zhang, Shiliang and Yang, Ming and Shan, Yanhu and Huang, Tiejun},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
pages = {3828--3837},
title = {Bi-Directional Cascade Network for Perceptual Edge Detection},
year = {2019}
}
```

