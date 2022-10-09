# Semi-supervised Semantic Segmentation with Prototype-based Consistency Regularization

This project contains the implementation of our work for semi-supervised semantic segmentation:
    
> Semi-supervised Semantic Segmentation with Prototype-based Consistency Regularization,   
> Hai-Ming Xu, Lingqiao Liu, Qiuchen Bian and Zhen Yang,   
> *To be appeared in NeurIPS 2022*

## News
* [2022-10-09] Repo is created. Code will come soon.

## Abstract

Semi-supervised semantic segmentation requires the model to effectively propagate the label information from limited annotated images to unlabeled ones. A challenge for such a per-pixel prediction task is the large intra-class variation, i.e., regions belonging to the same class may exhibit a very different appearance even in the same picture. This diversity will make the label propagation hard from pixels to pixels. To address this problem, we propose a novel approach to regularize the distribution of within-class features to ease label propagation difficulty. Specifically, our approach encourages the consistency between the prediction from a linear predictor and the output from a prototype-based predictor, which implicitly encourages features from the same pseudo-class to be close to at least one within-class prototype while staying far from the other between-class prototypes. By further incorporating CutMix operations and a carefully-designed prototype maintenance strategy, we create a semi-supervised semantic segmentation algorithm that demonstrates superior performance over the state-of-the-art methods from extensive experimental evaluation on both Pascal VOC and Cityscapes benchmarks.
