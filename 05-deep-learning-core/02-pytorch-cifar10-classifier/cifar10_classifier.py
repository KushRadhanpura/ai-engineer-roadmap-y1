import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- 1. Data Loading and Preprocessing ---

def get_cifar10_loaders(batch_size=4):
    """
    Downloads the CIFAR-10 dataset and creates data loaders for training and testing.
    """
    # Define transformations for the training and test sets
    # We'll normalize the images to have values between -1 and 1
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Download and load the training data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    # Download and load the test data
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    # Define the classes in the dataset
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
               
    return trainloader, testloader, classes

# --- 2. Define the Convolutional Neural Network (CNN) ---

class Net(nn.Module):
    """
    A simple CNN for CIFAR-10 classification.
    The architecture is: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC -> ReLU -> FC -> ReLU -> FC
    """
    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channels (RGB), 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        # 6 input channels from conv1, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 5x5 is the image dimension after 2 pools
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 output classes

    def forward(self, x):
        # Forward pass through the layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5) # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- 3. Training the Network ---

def train_network(net, trainloader, epochs=2, learning_rate=0.001):
    """
    Trains the neural network.
    """
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    print("--- Starting Training ---")
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print("--- Finished Training ---")

# --- 4. Main Execution Block ---

if __name__ == "__main__":
    # Get data loaders
    train_loader, test_loader, class_names = get_cifar10_loaders(batch_size=4)
    
    # Create the network
    cnn_net = Net()
    
    # Train the network
    # Note: Training for only 2 epochs is for demonstration. 
    # For good performance, you'd need more epochs.
    train_network(cnn_net, train_loader, epochs=2)
    
    # (Optional) A full implementation would also include an evaluation step
    # on the test set to check the model's accuracy.
    
    # --- Saving the trained model weights ---
    PATH = './cifar_net.pth' # .pth is the standard extension for PyTorch state dictionary
    torch.save(cnn_net.state_dict(), PATH)
    print(f'\nModel weights saved to {PATH}')

    # --- Evaluation Step: Test the model on the test data ---
    print('\n--- Starting Evaluation on Test Data ---')

    # Set the model to evaluation mode (important for certain layers like dropout/batch norm)
    cnn_net.eval() 

    correct = 0
    total = 0

    # Since we're not training, we don't need to calculate gradients
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            
            # Forward pass: calculate the model's output (predictions)
            outputs = cnn_net(images)
            
            # Get the predicted class (the index with the highest probability)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate and print the final accuracy
    accuracy = 100 * correct / total
    print(f'Finished Evaluation.')
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')
