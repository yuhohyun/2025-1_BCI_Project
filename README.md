# STaRNet: A spatio-temporal and Riemannian network for high-performance motor imagery decoding
This repository is an **unofficial PyTorch implementation** of the paper "**STaRNet: A spatio-temporal and Riemannian network for high-performance motor imagery decoding**". For more details, please refer to [paper link](https://www.sciencedirect.com/science/article/pii/S0893608024003952).

## Update!
**25.06.11:** Our repository was created!

## Overview

![STaRNet Architecture](figure/architecture.png)

**STaRNet** is a deep learning model that combines multi-scale feature extraction and Riemannian geometric features for high-performance motor imagery decoding. The model captures multi-scale temporal patterns from EEG signals and projects them onto the Riemannian manifold through covariance matrices. These features are then mapped to the tangent space and effectively fused for classification. The architecture demonstrates superior performance in decoding motor imagery tasks, particularly in the **BCI Competition IV 2a** dataset.

## Prerequisites

* **Ubuntu 22.04 (WSL2 possible)**
* Docker
* **CUDA 11.6.2**
* Python 3.10.8

## Installation

1.  **Clone Repository**
    ```bash
    git clone https://github.com/yuhohyun/2025-1_BCI_Project.git
    cd 2025-1_BCI_Project
    ```

2.  **Pull Docker Image**
    Run the following command in your terminal to pull the PyTorch development environment image from Docker Hub.
    ```bash
    docker pull pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
    ```

3.  **Run Docker Container**
    Start a container from the pulled image. This command mounts the current project folder into the container's `/workspace` directory and enables GPU access.
    ```bash
    docker run -it --rm --gpus all \
    -v "$(pwd)":/workspace -w /workspace \
    pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel /bin/bash
    ```

4.  **Install Requirements**
    Inside the running container, install the required Python packages listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

* We downloaded the dataset from the [link](https://www.bbci.de/competition/iv/download/index.html?agree=yes&submit=Submit).
* **Currently, we are providing the converted dataset.**
  
  → **So you don't need to download the dataset as we are providing the converted version.**

## Train

To train the model, run the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py
```

You can specify the GPU number by changing the number after `CUDA_VISIBLE_DEVICES=`. For example, to use GPU 7, run:
```bash
CUDA_VISIBLE_DEVICES=7 python main.py
```

**You can modify the hyperparameters in `config/params.py` before training. The training results will be saved in the path specified by the `--save_root` argument in `main.py`**.

## Evaluation
* We conducted our experiments in an NVIDIA RTX A6000.

**Table 1. Average accuracy and kappa values across STaRNet variants on the BCI Competition IV 2a dataset**

| Model            | Subject 1 | Subject 2 | Subject 3 | Subject 4 | Subject 5 | Subject 6 | Subject 7 | Subject 8 | Subject 9 | Avg. acc.        | Avg. kappa       |
|---------------------|-----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|-----------------:|-----------------:|
| **STaRNet (paper)**       |     88.54 |     69.10 |     94.10 |     82.29 |     78.12 |     70.83 |     93.40 |     88.54 |     84.72 | 83.29 (0.00)    | 0.777 (0.000)   |

**Experiment result**
| Model               | Subject 1 | Subject 2 | Subject 3 | Subject 4 | Subject 5 | Subject 6 | Subject 7 | Subject 8 | Subject 9 | Avg. acc.        | Avg. kappa       |
|---------------------|-----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|-----------------:|-----------------:|
| **STaRNet (ours)**       |     90.63 |     71.53 |     93.4 |     77.08 |     69.79 |     62.5 |     92.01 |     90.97 |     77.43 | 80.59 (−2.70)    | 0.737(-0.040)   |

## License

This repository is released under the MIT license as found in the [LICENSE](https://github.com/ddongpal/2025-1-bci-project/blob/main/LICENSE) file.   

## Acknowledgement

* I would like to express my deepest gratitude to **Professor Byung Hyung Kim**, who guided this project, and **Teaching Assistant Li Hanyu**, who provided meticulous assistance!
* Special thanks to the authors of the original paper: **Xingfu Wang, Wenjie Yang, Wenxia Qi, Yu Wang, Xiaojun Ma, and Wei Wang**, for their groundbreaking work on **STaRNet**.
