# Semi-supervised Semantic Segmentation with Prototype-based Consistency Regularization

This project contains the implementation (based on the MindSpore framework) of our work for semi-supervised semantic segmentation:
    
> Semi-supervised Semantic Segmentation with Prototype-based Consistency Regularization,   
> Hai-Ming Xu, Lingqiao Liu, Qiuchen Bian and Zhen Yang,   
> *Accepted to NeurIPS 2022*

## Abstract

Semi-supervised semantic segmentation requires the model to effectively propagate the label information from limited annotated images to unlabeled ones. A challenge for such a per-pixel prediction task is the large intra-class variation, i.e., regions belonging to the same class may exhibit a very different appearance even in the same picture. This diversity will make the label propagation hard from pixels to pixels. To address this problem, we propose a novel approach to regularize the distribution of within-class features to ease label propagation difficulty. Specifically, our approach encourages the consistency between the prediction from a linear predictor and the output from a prototype-based predictor, which implicitly encourages features from the same pseudo-class to be close to at least one within-class prototype while staying far from the other between-class prototypes. By further incorporating CutMix operations and a carefully-designed prototype maintenance strategy, we create a semi-supervised semantic segmentation algorithm that demonstrates superior performance over the state-of-the-art methods from extensive experimental evaluation on both Pascal VOC and Cityscapes benchmarks.

## Results
### PASCAL VOC 2012

**classic** setting: Labeled images are selected from the ```train``` set of original VOC, ```1,464``` images in total and the remaining ```9,118``` images are all considered as unlabeled ones.

| Method                      | 1/16 (92) | 1/8 (183) | 1/4 (366) | 1/2 (732) | Full (1464) |
| --------------------------- | --------- | --------- | --------- | --------- | ----------- |
| Supervised Only             | 45.77     | 54.92     | 65.88     | 71.69     | 72.50       |
| U<sup>2</sup>PL             | 67.98     | 69.15     | 73.66     | 76.16     | 79.49       |
| **Ours (paper)**            | 70.06     | 74.71     | 77.16     | 78.49     | 80.65       |
| **Ours (this repo.)**       | 70.78     | 74.65     | 76.77     | 78.70     | 80.18       |


### Cityscapes

Labeled images are selected from the ```train``` set, ```2,975``` images in total. 

| Method                      | 1/16 (186) | 1/8 (372) | 1/4 (744) | 1/2 (1488) |
| --------------------------- | ---------- | --------- | --------- | ---------- |
| Supervised Only             | 65.74      | 72.53     | 74.43     | 77.83      |
| U<sup>2</sup>PL             | 70.30      | 74.37     | 76.47     | 79.05      |
| **Ours (paper)**            | 73.41      | 76.31     | 78.40     | 79.11      |
| **Ours (this repo.)**       | 73.99      | 77.10     | 78.58     | 79.55      |

> Since the current MindSpore version is transfered from the checkpoint trained with the code implemented in Pytorch, inconsistent operator interfaces may lead to different precisions.

## Checkpoints

Checkpoints for VOC and CityScapes can be downloaded from the [google driver](https://drive.google.com/drive/folders/118LBQpXO9m8zr1mjTlNf_lnRI_3Y4BER?usp=sharing) and the folder orgaization is as follows

```shell
.
└──ckpt
  ├── city
    ├── 183
      └── semiseg_with_proto
        └── checkpoints
          └──best.ckpt
    ├── 372
      └── semiseg_with_proto
        └── checkpoints
          └──best.ckpt
    ├── 744
      └── ...
    └── 1488
      └── ...
  ├── voc
    ├── 92
      └── semiseg_with_proto
        └── checkpoints
          └──best.ckpt
    ├── 183
      └── semiseg_with_proto
        └── checkpoints
          └──best.ckpt
    ├── 366
      └── ...
    ├── 732
      └── ...
    └── 1464
      └── ...
```

## Installation

The running environment is almost identical to the [U<sup>2</sup>PL codebase](https://github.com/Haochen-Wang409/U2PL/#installation) and we further install the following lib for running the MindSpore framework

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.10.1/MindSpore/gpu/x86_64/cuda-10.1/mindspore_gpu-1.10.1-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> Please note that the mindspore 1.10.1 requires the cuda 10.1 and the cuDNN 7.6.x or 8.0.x.

## Evaluation

The config files locate at `experiments` folder, please enter correspoinding folders for evaluation：

```bash
cd experiments/cityscapes/186/ours
bash eval.sh
```

## Acknowledgement

We thank the following [U<sup>2</sup>PL codebase](https://github.com/Haochen-Wang409/U2PL/#installation) and [AEL](https://github.com/hzhupku/SemiSeg-AEL) for their impressive work and open-sourced projects.


## Citation
```bibtext
@inproceedings{xu2022semisupervised,
    title={Semi-supervised Semantic Segmentation with Prototype-based Consistency Regularization},
    author={Hai-Ming Xu and Lingqiao Liu and Qiuchen Bian and Zhen Yang},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2022},
}
```