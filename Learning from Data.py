import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
import seaborn as sns
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms with more augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Training images
# Load datasets
trainset = torchvision.datasets.ImageFolder(
    root=r'C:\Users\omerf\OneDrive\Masaüstü\Learning From Data Project\classes_train', transform=transform)
testset = torchvision.datasets.ImageFolder(
    root=r'C:\Users\omerf\OneDrive\Masaüstü\Learning From Data Project\classes_test', transform=transform)

# Creating sampler for class balance
class_counts = [len([label for _, label in trainset if label == i]) for i in range(10)]
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [0] * len(trainset)

for idx, (data, label) in enumerate(trainset):
    class_weight = class_weights[label]
    sample_weights[idx] = class_weight

sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

loader_train = torch.utils.data.DataLoader(trainset, batch_size=64, sampler=sampler, num_workers=8)
loader_test = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)

# Define the embedding model with dropout
class EmbeddingModel(nn.Module):
    def __init__(self, base_model, embedding_size):
        super(EmbeddingModel, self).__init__()
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(base_model.fc.in_features, embedding_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)
        return x

# Load pretrained ResNet18 and create embedding model
resnet = torchvision.models.resnet18(pretrained=True)
embedding_resnet = EmbeddingModel(resnet, embedding_size=128).to(device)


# Define contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

criterion_contrastive = ContrastiveLoss()
optimizer_contrastive = optim.Adam(embedding_resnet.parameters(), lr=0.00001)

# Function to create pairs
def create_pairs(images, labels, num_pairs=16):
    pairs = []
    for i in range(num_pairs):
        idx1, idx2 = torch.randint(0, len(images), (2,))
        img1, img2 = images[idx1], images[idx2]
        label = torch.tensor(int(labels[idx1] == labels[idx2]), dtype=torch.float32)
        pairs.append((img1, img2, label))
    return pairs

# Training loop for the contrastive loss
losses = []
num_epochs_contrastive = 20
for epoch in range(num_epochs_contrastive):
    embedding_resnet.train()
    running_loss = 0.0
    for images, labels in loader_train:
        images, labels = images.to(device), labels.to(device)
        pairs = create_pairs(images, labels, num_pairs=16)
        for img1, img2, label in pairs:
            optimizer_contrastive.zero_grad()
            img1, img2, label = img1.unsqueeze(0).to(device), img2.unsqueeze(0).to(device), label.to(device)

            output1 = embedding_resnet(img1)
            output2 = embedding_resnet(img2)

            loss = criterion_contrastive(output1, output2, label)

            loss.backward()
            optimizer_contrastive.step()

            running_loss += loss.item()

    avg_loss = running_loss / len(loader_train)
    print(f"Epoch: {epoch}, Contrastive Loss: {avg_loss:.4f}")
    losses.append(avg_loss)

# Plotting the loss
sns.lineplot(x=range(len(losses)), y=losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Loss Change in Every Epoch")
plt.show()

# Adding linear classifier on top of the Embedding Model with BatchNorm
classifier = nn.Sequential(
    nn.Linear(in_features=128, out_features=512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Linear(in_features=512, out_features=10)
).to(device)


class FullModel(nn.Module):
    def __init__(self, embedding_model, classifier):
        super(FullModel, self).__init__()
        self.embedding_model = embedding_model
        self.classifier = classifier

    def forward(self, x):
        x = self.embedding_model(x)
        x = self.classifier(x)
        return x

# Create full model
full_model = FullModel(embedding_resnet, classifier).to(device)
# %%
# Fine-tuning additional layers
for param in full_model.embedding_model.base_model[-3:].parameters():
    param.requires_grad = True
# Loss and optimizer for the classifier
criterion_classifier = nn.CrossEntropyLoss()
optimizer_classifier = optim.SGD(full_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer_classifier, step_size=7, gamma=0.1)

# Training loop for the classifier
num_epochs_classifier = 10
loss1 = []
accuracy = []

for epoch in range(num_epochs_classifier):
    full_model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader_train:
        images, labels = images.to(device), labels.to(device)
        optimizer_classifier.zero_grad()

        outputs = full_model(images)
        loss = criterion_classifier(outputs, labels)

        loss.backward()
        optimizer_classifier.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(loader_train)
    acc = 100 * correct / total
    print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
    loss1.append(avg_loss)
    accuracy.append(acc)

    scheduler.step()

# Plotting the classifier loss
sns.lineplot(x=range(len(loss1)), y=loss1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Classifier Loss Change in Every Epoch")
plt.show()

# Plotting the classifier accuracy
sns.lineplot(x=range(len(accuracy)), y=accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Classifier Accuracy Change in Every Epoch")
plt.show()

# Test Phase
test_losses = []
test_accuracy = []
epoch_test = 10
for epoch in range(epoch_test):
    full_model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader_test:
            images, labels = images.to(device), labels.to(device)
            outputs = full_model(images)
            loss = criterion_classifier(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(loader_test)
    test_losses.append(avg_test_loss)
    test_acc = 100 * correct / total
    test_accuracy.append(test_acc)

    print(f"Epoch: {epoch},Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

# Plotting the train and test loss
plt.figure(figsize=(12, 6))
sns.lineplot(x=range(len(test_losses)), y=test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Test Loss Change in Every Epoch")
plt.legend()
plt.show()

# Plotting the train and test accuracy
plt.figure(figsize=(12, 6))
sns.lineplot(x=range(len(test_accuracy)), y=test_accuracy, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Accuracy Change")
plt.legend()
plt.show()