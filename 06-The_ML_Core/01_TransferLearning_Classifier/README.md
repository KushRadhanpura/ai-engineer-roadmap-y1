# Niche Image Classifier: Rock, Paper, Scissors

This project demonstrates the use of transfer learning to train a ResNet50 model to classify images of rock, paper, and scissors.

## Project Structure

- `data/`: Contains the processed image data, split into `train` and `val` sets.
- `models/`: Stores the trained model weights.
- `notebooks/`: Contains the Jupyter Notebook used for data exploration and initial model development.
- `src/`: Contains the Python scripts for the data loading, training, and prediction logic.
  - `dataset.py`: Defines the data loading and transformation pipeline.
  - `train.py`: The main training script.
  - `predict.py`: A script to run inference on new images.

## Methodology

1.  **Data Preparation**: The dataset is organized into `train` and `val` directories, with subdirectories for each class (rock, paper, scissors).
2.  **Data Transformation**: PyTorch's `torchvision.transforms` is used to apply data augmentation to the training set and normalization to both the training and validation sets.
3.  **Transfer Learning**: A pre-trained ResNet50 model is loaded, and all layers are frozen except for the final classification layer.
4.  **Model Training**: The new classification layer is trained on the rock-paper-scissors dataset.
5.  **Inference**: The trained model is used to predict the class of new images.

## Results

The model was trained for 10 epochs. While the training loss decreased, the validation accuracy remained low, indicating that the model did not generalize well to the validation set.

**Training Output:**
```
Starting training...
Epoch 1/10, Loss: 0.7825
Validation Loss: 1.8990, Accuracy: 0.0857
Epoch 2/10, Loss: 0.4452
Validation Loss: 2.1246, Accuracy: 0.1429
Epoch 3/10, Loss: 0.3864
Validation Loss: 2.6918, Accuracy: 0.1143
Epoch 4/10, Loss: 0.3214
Validation Loss: 3.3778, Accuracy: 0.0857
Epoch 5/10, Loss: 0.3173
Validation Loss: 2.6672, Accuracy: 0.2000
Epoch 6/10, Loss: 0.3063
Validation Loss: 3.6970, Accuracy: 0.1429
Epoch 7/10, Loss: 0.2745
Validation Loss: 3.4351, Accuracy: 0.1714
Epoch 8/10, Loss: 0.2529
Validation Loss: 4.0437, Accuracy: 0.1429
Epoch 9/10, Loss: 0.2508
Validation Loss: 3.4871, Accuracy: 0.1714
Epoch 10/10, Loss: 0.2337
Validation Loss: 5.9443, Accuracy: 0.0571
Training complete.
Model saved to /home/kushsoni/Desktop/ai-engineer-roadmap-y1/06-The_ML_Core/01_TransferLearning_Classifier/models/rock_paper_scissors_resnet50.pth
```

**Prediction Output:**
```
The model predicts the image 'rock01-000.png' is: rock
The model predicts the image 'paper-hires1.png' is: scissors
```

## Analysis of Results

The low validation accuracy suggests several potential issues:
*   **Data Imbalance**: The validation set was very small and may not have been representative of the overall data distribution.
*   **Insufficient Training**: 10 epochs may not have been enough for the model to learn the features of the dataset.
*   **Learning Rate**: The learning rate may have been too high, causing the model to overshoot the optimal weights.
*   **Overfitting**: The model may have overfit to the training data, as indicated by the decreasing training loss and fluctuating validation loss.

## Future Steps

To improve the model's performance, the following steps could be taken:
*   **Increase the size of the validation set.**
*   **Train for more epochs.**
*   **Experiment with different learning rates.**
*   **Apply more aggressive data augmentation techniques.**
