# Project Overview

This repository contains the implementation of the U-Net architecture for image segmentation using both PyTorch and TensorFlow frameworks. The project includes scripts for model training, utilities for running the models, and configurations for setting up the environment via Docker.

## File Descriptions

- **CNN_pytorch.py**: This script implements the U-Net architecture using PyTorch. It includes the model definition, training loop, and inference functionalities. The code is structured for ease of understanding and can be modified for custom datasets.

- **CNN_tensorflow.py**: Similar to the PyTorch implementation, this script implements the U-Net architecture using TensorFlow. It follows the same architecture principles and allows for model training and inference.

- **menu.sh**: A shell script that provides a user-friendly menu interface for interacting with this project. Users can easily choose options to run training, inference, or other tasks without needing to remember specific command-line inputs.

- **requirements.txt**: This file lists all the necessary Python packages and their versions required to run the scripts. It can be used to set up the Python environment using pip.

- **Dockerfile**: This file contains the instructions to build a Docker image that encapsulates all the dependencies required for the project. It provides a consistent environment, which helps eliminate issues related to package versions and system configurations.

- **docker-compose.yml**: This file is used to configure the services required for running the Docker containers. It simplifies the process of starting multiple services, allowing users to run the application with a single command.

## Overall Project Structure

```
CNN_U-Net/
├── CNN_pytorch.py
├── CNN_tensorflow.py
├── menu.sh
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

This structure keeps the project organized and accessible for users to navigate through various scripts and configurations easily.

---

*Created on 2026-02-15 22:14:40 UTC*