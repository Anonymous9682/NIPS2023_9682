# ReadMe

## Overview

 Official pytorch implementation of the paper:

<!-- ["*CLIP as Multi-Task Multi-Kernel Learning*" (2023)](https://openreview.net/forum?id=FAkiXRVxjX) [Yucong Lin](https://openreview.net/profile?id=~Yucong_Lin2), [Tianjun Ke](https://openreview.net/profile?id=~Tianjun_Ke1), [Xingpeng Xia](https://openreview.net/profile?id=~Xingpeng_Xia1), [Jiaheng Yin](https://openreview.net/profile?id=~Jiaheng_Yin1), [Jiaxing Xu](https://openreview.net/profile?id=~Jiaxing_Xu1), [Tianxi Cai](https://openreview.net/profile?id=~Tianxi_Cai1), [Junwei Lu](https://openreview.net/profile?id=~Junwei_Lu1) -->

["*CLIP as Multi-Task Multi-Kernel Learning*" (2023)](https://openreview.net/forum?id=FAkiXRVxjX)

### Description

We provide a theoretical interpretation of CLIP utilizing the *Reproducing Kernel Hilbert Space (RKHS)*  framework. Specifically, we reformulate the problem of estimating the infinite-dimensional mapping $\phi$ with a neural network as selecting an unknown RKHS using multiple kernel learning. Such connection motivates us to propose to estimate the CLIP embedding via the multi-task multi-kernel (MTMK) method: we reformulate the different labels in the CLIP training data as the multiple training tasks, and reformulate learning the unknown CLIP embedding as choosing an optimal kernel from a family of Reproducing Kernel Hilbert Spaces, which is computationally more efficient. 
Utilizing the MTMK interpretation of CLIP, we also show an optimal statistical rate of the MTMK classifier under the scenario that both the number of covariates and the number of candidate kernels can increase with the sample size. Besides the synthetic simulations, we apply the proposed method to align the medical imaging data with the clinical codes in electronic health records and illustrate that our approach can learn the proper kernel space aligning the imaging embedding with the text embeddings with high accuracy.

### Requirements

1. Python >= 3.8
2. Numpy >= 1.17
3. [pyTorch](https://pytorch.org/) >= 1.2.0
4. [GPyTorch](https://gpytorch.ai/) >= 0.3.5
5. sklearn >= 0.24.2
6. pandas >= 1.3.0

## Experiments

### Dataset

```shell
cd ./data
```

`organcmnist.npz` is OrganCMNIST dataset, part of the MedMNIST collection. It consists of grayscale images of abdominal CT scans that depict $11$ different organs, including the bladder, femurs, and nine other organs. 

`phecode_final.tsv` includes all the word embeddings used in our experiments and their corresponding descriptions.

### Methods

We implement MTMK and STMK model classes in `mt_model_gpu_grad.py`. MTSK and STSK can be derived from them, for example, we can simply choose only one kernel in our kernel list and it is exactly MTSK/STSK. Additionally, we use $\lambda_2,\lambda_3$ to control the norm of our model.

### Training and Testing Procedure

We simply select 7 labels from MedMNIST dataset and the model can be trained and tested by 5-fold cross validation as follows:

e.g. `python MedMNIST.py --method="MTMK mix" --seed=53 --samples=100 --texts=0 --lmd2=0.05 --lmd3=0.05`

Here is the description of all parameters:

* --method: choose a model from *MTMK mix/ MTMK L1/ MTMK L2/ LR/ SVM/ STMK/ STSK*
* --seed: the random seed used in selecting picture and cross validation. Default value is 53.
* --samples: the number of positive images in each task, i.e. if sample=100, then the number of positive and negative images are all 100, so there will be 200 images in one task.
* --text: the number of positive text embeddings in each task.
* --lmd2, --lmd3: parameters of our model.

### Tuning parameters($\lambda_2,\lambda_3$) 

The parameters can be tuned with grid search, for example, if we want to tune ($\lambda_2,\lambda_3$)  on a grid of $[0.005,0.05]\times[0.005,0.05]$ with a step=0.005, the code is as follows:

`python MedMNIST_parameter_tune.py --method="MTMK mix" --seed=53 --samples=100 --texts=50 --lmd2=0.005 --lmd3=0.005 --start=0.005 --end=0.05 --step=0.005`

Then the best parameters will be printed in the console and also the best results of auc will be saved as `.csv` documents.

### Result

* **Auc**: After each training(except the process of grid search), the results will be automatically saved into `result` documentary.
* **Activation Maps**: `Activation Maps.ipynb` contains the code for the heat map picture in our paper, which reveals that our model consistently extracts similar features for each organ.
* **Inner Products Maps**: In `Extract_inner_product.ipynb` , we give an example notebook on how to extract weight vector via NMF and compute inner product.
