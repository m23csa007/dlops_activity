import torch
import torchvision
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Check last digit of your roll number
last_digit = 7

if last_digit % 3 == 0:
    dataset_name = 'STL10'
elif last_digit % 3 == 1:
    dataset_name = 'SVHN'
else:
    dataset_name = 'FashionMNIST'

# Define transforms for the dataset
transform = transforms.Compose([
    transforms.Resize(224),  # Resize to match input size of ResNet101
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load dataset
if dataset_name == 'STL10':
    train_dataset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
    test_dataset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)
    num_classes = 10
elif dataset_name == 'SVHN':
    train_dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    num_classes = 10
else:  # FashionMNIST
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    num_classes = 10

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load pre-trained ResNet101
model = torchvision.models.resnet101(pretrained=True)

# Replace the final classification layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Define optimizer options
optimizers = ['Adam', 'Adagrad', 'RMSprop']
criterion = nn.CrossEntropyLoss()

# Training loop
def train_model(optimizer_name):
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    else:
        raise ValueError("Invalid optimizer name")

    epochs = 10
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()  # Move data to GPU if available
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

    return train_losses, train_accuracies

# Train with each optimizer
for optimizer_name in optimizers:
    print(f"\nTraining with {optimizer_name} optimizer:")
    model.cuda()  # Move model to GPU if available
    train_losses, train_accuracies = train_model(optimizer_name)

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.title(f'Training Curve with {optimizer_name} Optimizer')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Test model
model.eval()
top5_correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()  # Move data to GPU if available
        outputs = model(images)
        _, predicted = torch.topk(outputs, 5, dim=1)
        for i in range(labels.size(0)):
            if labels[i] in predicted[i]:
                top5_correct += 1
        total += labels.size(0)

top5_accuracy = top5_correct / total
print(f"\nFinal Top-5 Test Accuracy: {top5_accuracy:.4f}")
