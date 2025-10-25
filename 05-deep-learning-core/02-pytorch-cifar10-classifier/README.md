# Project: CIFAR-10 Image Classifier with PyTorch

This project implements a simple Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The entire implementation is done using the PyTorch framework.

## Description

The goal of this project is to build a foundational understanding of how to construct, train, and evaluate a neural network for a computer vision task using PyTorch.

The script `cifar10_classifier.py` handles the following:
1.  **Data Loading**: Downloads the CIFAR-10 dataset and prepares it for training and testing using `torchvision`.
2.  **Data Preprocessing**: Applies transformations to the image data, including converting them to tensors and normalizing them.
3.  **Model Definition**: Defines a simple CNN architecture with convolutional, pooling, and fully connected layers.
4.  **Training Loop**: Implements a training loop that feeds data to the model, calculates the loss (Cross-Entropy), and updates the model's weights using an optimizer (SGD).

## How to Run

1.  **Install Dependencies**:
    Make sure you have PyTorch and TorchVision installed in your Python environment.
    ```bash
    pip install torch torchvision
    ```

2.  **Run the Script**:
    Execute the script from your terminal.
    ```bash
    python cifar10_classifier.py
    ```
    The script will automatically download the CIFAR-10 dataset into a `data` directory and start the training process.

## Next Steps

- Implement an evaluation function to measure the model's accuracy on the test set.
- Experiment with different hyperparameters (e.g., learning rate, batch size, number of epochs).
- Modify the CNN architecture (e.g., add more layers, use different activation functions) to improve performance.
