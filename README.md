# STaRNet: A spatio-temporal and Riemannian network for high-performance motor imagery decoding
This repository is an **unofficial PyTorch implementation** of the paper "**STaRNet: A spatio-temporal and Riemannian network for high-performance motor imagery decoding**".

## Overview

## STaRNet

## Evaluation

## Dataset Download

## Prerequisites

* **Ubuntu 22.04 (WSL2 possible)**
* **Docker**

---

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

---
## License
This repository is released under the MIT license as found in the [LICENSE](https://github.com/ddongpal/2025-1-bci-project/blob/main/LICENSE) file.   
## Acknowledgement

