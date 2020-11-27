# Maximum-Entropy Adversarial Data Augmentation for Improved Generalization and Robustness (ME-ADA)
This repository contains the Pytorch implementation of [Maximum-Entropy Adversarial Data Augmentation for Improved Generalization and Robustness](https://arxiv.org/abs/2010.08001). If you find our code useful in your research, please cite:

```
@inproceedings{zhaoNIPS20maximum,
  author    = {Zhao, Long and Liu, Ting and Peng, Xi and Metaxas, Dimitris},
  title     = {Maximum-Entropy Adversarial Data Augmentation for Improved Generalization and Robustness},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2020}
}
```

## Quick start

This repository reproduces our results on MNIST and CIFAR-10, which is build upon Python v2.7 and Pytorch v1.1.0 on Ubuntu 16.04 (other dependencies include: `numpy`, `scipy`, and `scikit-learn`). The code may also work with Python v3 but has not been tested. NVIDIA GPUs are needed to train and test. We recommend installing Python v2.7 from [Anaconda](https://www.anaconda.com/), and installing Pytorch (>= 1.1.0) following guide on the [official instructions](https://pytorch.org/) according to your specific CUDA version.

Then you can clone this repository with the following commands:

```
git clone git@github.com:garyzhao/ME-ADA.git
cd ME-ADA
```

### Results on MNIST

To reproduce the result on MNIST, please follow the steps as below:

1. Run the command to create the `data` folder if it does not exist:
    ```
    mkdir data
    ```

2. Download the MNIST-M dataset from [https://drive.google.com/drive/folders/0B_tExHiYS-0vR2dNZEU4NGlSSW8](https://drive.google.com/drive/folders/0B_tExHiYS-0vR2dNZEU4NGlSSW8), rename the folder by `MNIST_M`, and move it to the `data` folder.

3. Download the SYN dataset from [https://drive.google.com/file/d/0B9Z4d7lAwbnTSVR1dEFSRUFxOUU/view](https://drive.google.com/file/d/0B9Z4d7lAwbnTSVR1dEFSRUFxOUU/view), rename the folder by `SYN`, and move it to the `data` folder.

4. Run the command:
    ```
    sh run_main_mnist.sh
    ```

5. The results will be stored in the `mnist` folder. 

### Results on CIFAR-10

To reproduce the result on CIFAR-10, please follow the steps as below:

1. Run the command to create the `data` folder if it does not exist:
    ```
    mkdir data
    ```
   
2. Download the CIFAR-10-C dataset from [https://zenodo.org/record/2535967/files/CIFAR-10-C.tar](https://zenodo.org/record/2535967/files/CIFAR-10-C.tar), rename the folder by `CIFAR-10-C`, and move it to the `data` folder.

3. Run the command:
    ```
    sh run_main_cifar10.sh
    ```
   
4. The results will be stored in the `cifar10` folder.

Please find the test accuracy in `best_test.txt` for each run. You can try different algorithms (ERM, ADA, and ME-ADA) by modifying the `--algorithm` parameter in the script. To use different network architectures (AllConvNet, DenseNet, WideResNet, and ResNeXt) on CIFAR-10, please change the `--model` parameter in `run_main_cifar10.sh`.

## Acknowledgement

Part of our code is borrowed from the following repositories.

- [M-ADA](https://github.com/joffery/M-ADA): "Learning to Learn Single Domain Generalization", CVPR 2020.
- [Episodic-DG](https://github.com/HAHA-DL/Episodic-DG): "Episodic Training for Domain Generalization", ICCV 2019.
- [AugMix](https://github.com/google-research/augmix): "AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty", ICLR 2020.

We thank to the authors for releasing their codes. Please also consider citing their works.