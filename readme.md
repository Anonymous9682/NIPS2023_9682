# ReadMe

## Data

The original MedMNIST image data are `organcmnist.npz`, and `phecode_final.tsv` stores all the embeddings of medical descriptions.

## Result

Contains all the results in the table that is shown in the table of our paper.

## Codes

1. `kernels_gpu_grad.py`: class of several kernel functions, including Gaussian kernel, linear and polynomial kernel
2. `MedMNIST_parameter_tune.py`: tune the parameters of $\lambda_2$ and $\lambda_3$ of several models with grid search and automatically save the result of the best model. `run.sh` gives an example to use it.
3.  `mt_model_gpu_grad.py`: implementation of our model(MTMK), and offering all the interface of MTMK



