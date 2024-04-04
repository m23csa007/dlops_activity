import torch.nn.functional as F

# Define MLP Model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(16*16, 512*2)
        self.fc2 = nn.Linear(512*2, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.fc1 = nn.Linear(64 * 64 * 4, 128)  # Corrected input size
#         self.fc2 = nn.Linear(128, 64)

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.max_pool2d(x, 2, 2)
#         x = torch.relu(self.conv2(x))
#         x = torch.max_pool2d(x, 2, 2)
#         #x = x.view(-1, 16384)  # Corrected view size
#         x = x.view(-1, 64 * 7 * 7)  # Corrected view size based on feature map dimensions after max pooling

#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels = 1, out_channels = 16, padding= 3, kernel_size = 7)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2)
        self.conv_layer2 = nn.Conv2d(in_channels = 16, out_channels = 8, padding= 2, kernel_size = 5)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2)
        self.conv_layer3 = nn.Conv2d(in_channels = 8, out_channels = 4, padding= 1, kernel_size = 3)
        self.avg_pool1= nn.AdaptiveAvgPool2d(2)

        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(4*4*4, num_classes)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.relu1(self.conv_layer1(x))
        out = self.max_pool1(out)
        out = self.relu1(self.conv_layer2(out))
        out = self.max_pool2(out)

        out = self.relu1(self.conv_layer3(out))
        out = self.avg_pool1(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.Softmax(out)

        return out

# Define evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return accuracy, recall, precision, conf_matrix




# Train MLP
mlp_model = MLP()
mlp_model.to(device)
mlp_criterion = nn.CrossEntropyLoss()
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.1)

num_epochs = 1
writer = SummaryWriter('logs/mlp')

for epoch in range(num_epochs):
    # Training loop
    mlp_model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        mlp_optimizer.zero_grad()
        outputs = mlp_model(inputs)
        loss = mlp_criterion(outputs, labels)
        loss.backward()
        mlp_optimizer.step()

    # Evaluate on test set
    accuracy, recall, precision, conf_matrix = evaluate_model(mlp_model.to(device), test_loader)
    writer.add_scalar('MLP/Loss', loss.item(), epoch)
    writer.add_scalar('MLP/Accuracy', accuracy, epoch)
    writer.add_scalar('MLP/Recall', recall, epoch)
    writer.add_scalar('MLP/Precision', precision, epoch)