# cnn_classifier.py (Project 03)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# --- 1. Data Preparation (Same as Project 02) ---
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --- 2. Define the CNN Model (The Core Change) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1st Convolutional Layer
        # Input: 3 color channels (RGB)
        self.conv1 = nn.Conv2d(3, 6, 5) # Output: 6 feature maps
        self.pool = nn.MaxPool2d(2, 2)
        
        # 2nd Convolutional Layer
        self.conv2 = nn.Conv2d(6, 16, 5) # Output: 16 feature maps
        
        # Fully Connected Layers (Input must match the flattened size after conv/pool)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 16*5*5 is calculated from CIFAR-10's 32x32 image size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 classes

    def forward(self, x):
        # Apply Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))
        # Apply Conv2 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        # Apply Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = SimpleCNN()

# --- 3. Training and Evaluation (Same Logic as Project 02) ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("\n--- Starting CNN Training (More epochs recommended) ---")

# We will train for 5 epochs this time, as CNNs take longer to learn
for epoch in range(5): 
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training.')

# --- 4. Final Evaluation ---
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the CNN on the 10000 test images: {accuracy:.2f} %')

# --- 5. Save the Final Model ---
PATH = './cifar_cnn_net.pth'
torch.save(net.state_dict(), PATH)
print(f'Model weights saved to {PATH}')