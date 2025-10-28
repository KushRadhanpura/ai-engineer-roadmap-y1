import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import json

def predict(image_path):
    """
    Loads the trained model and predicts the class of an input image.
    """
    # --- 1. Load Model and Class Names ---
    
    # Define the model architecture (must match the one used for training)
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    # The number of classes must be the same as during training
    num_classes = 3 
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Get the absolute path to the model file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, "models", "rock_paper_scissors_resnet50.pth")

    # Load the saved model state dictionary
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Manually define class names based on the folder structure used for training
    class_names = ['paper', 'rock', 'scissors']

    # --- 2. Define Image Transformations ---
    
    # Use the same validation transforms as in dataset.py
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # --- 3. Process the Input Image ---

    # Open the image using PIL
    image = Image.open(image_path).convert('RGB')
    
    # Apply the transformations and add a batch dimension (C, H, W) -> (B, C, H, W)
    image_tensor = val_transforms(image).unsqueeze(0)

    # --- 4. Make a Prediction ---
    
    with torch.no_grad():
        outputs = model(image_tensor)
        # Get the index of the highest probability
        _, predicted_idx = torch.max(outputs, 1)
    
    # Map the index to the class name
    predicted_class = class_names[predicted_idx.item()]
    
    return predicted_class

if __name__ == '__main__':
    # Example usage:
    # You need to provide a path to an image you want to classify.
    # Let's use an image from the validation set for this example.
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Using a known image from the validation set to test
    # Note: The file 'rock01-000.png' was moved to the val set earlier
    test_image_path = os.path.join(base_dir, "data", "processed", "val", "rock", "rock01-000.png")

    if os.path.exists(test_image_path):
        prediction = predict(test_image_path)
        print(f"The model predicts the image '{os.path.basename(test_image_path)}' is: {prediction}")
    else:
        print(f"Test image not found at: {test_image_path}")
        print("Please ensure you have an image at that location to test the prediction.")

    # Example with another class
    test_image_path_paper = os.path.join(base_dir, "data", "processed", "val", "paper", "paper-hires1.png")
    if os.path.exists(test_image_path_paper):
        prediction = predict(test_image_path_paper)
        print(f"The model predicts the image '{os.path.basename(test_image_path_paper)}' is: {prediction}")

