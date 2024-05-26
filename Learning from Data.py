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
# Declaring Transforms (Resize and normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Training images
trainset = torchvision.datasets.ImageFolder(
    root=r'C:\Users\omerf\OneDrive\Masaüstü\Learning From Data Project\classes_train', transform=transform)
# Loading train images
loader_train = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8)
# Test images
testset = torchvision.datasets.ImageFolder(
    root=r'C:\Users\omerf\OneDrive\Masaüstü\Learning From Data Project\classes_test', transform=transform)
# Loading test images
loader_test = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)
# Loading ResNet18-Model
resnet = models.resnet18(pretrained=True)

# Creating a class Embedding Model to distinguish positive and negative samples
class EmbeddingModel(nn.Module):
    def __init__(self, base_model, embedding_size):
        super(EmbeddingModel, self).__init__()
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])  # Remove the last fully connected layer
        self.fc = nn.Linear(base_model.fc.in_features, embedding_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)
        return x

embedding_resnet = EmbeddingModel(resnet, embedding_size=128).to(device)


# Creating a class ContrastiveLoss to promote the attraction of similar pairs together (positive samples) and the removal of dissimilar pairs (negative samples)
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + label *
                                      torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


# Defining contrastive loss and optimizer
criterion_contrastive = ContrastiveLoss()
optimizer_contrastive = optim.Adam(embedding_resnet.parameters(), lr=0.0001)


# Function to create positive and negative pairs
def create_pairs(images, labels, num_pairs=16):
    pairs = []
    labels_list = labels.tolist()
    unique_labels = list(set(labels_list))

    for _ in range(num_pairs):
        # Positive pair
        pos_label = random.choice(unique_labels)
        pos_indices = [i for i, label in enumerate(labels_list) if label == pos_label]
        if len(pos_indices) > 1:
            i, j = random.sample(pos_indices, 2)
            pairs.append((images[i].unsqueeze(0), images[j].unsqueeze(0), torch.tensor([1.0], device=device)))

        # Negative pair
        neg_label1, neg_label2 = random.sample(unique_labels, 2)
        neg_index1 = random.choice([i for i, label in enumerate(labels_list) if label == neg_label1])
        neg_index2 = random.choice([i for i, label in enumerate(labels_list) if label == neg_label2])
        pairs.append(
            (images[neg_index1].unsqueeze(0), images[neg_index2].unsqueeze(0), torch.tensor([0.0], device=device)))

    return pairs

# Training loop for the contrastive loss
losses = []
num_epochs_contrastive = 10
for epoch in range(num_epochs_contrastive):
    embedding_resnet.train()
    running_loss = 0.0
    for images, labels in loader_train:
        images, labels = images.to(device), labels.to(device)
        pairs = create_pairs(images, labels, num_pairs=16)
        for img1, img2, label in pairs:
            optimizer_contrastive.zero_grad()

            output1 = embedding_resnet(img1)
            output2 = embedding_resnet(img2)

            loss = criterion_contrastive(output1, output2, label)

            loss.backward()
            optimizer_contrastive.step()

            running_loss += loss.item()

    avg_loss = running_loss / len(loader_train)
    print(f"Epoch: {epoch}, Contrastive Loss: {avg_loss}")
    losses.append(avg_loss)

sns.lineplot(x=range(len(losses)), y=losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Loss Change in Every Epoch")
plt.show()

# Adding linear classifier on top of the Embedding Model
classifier = nn.Sequential(
    nn.Linear(in_features=128, out_features=512),
    nn.ReLU(),
    nn.Linear(in_features=512, out_features=10)
)

embedding_resnet.fc = classifier.to(device)

# Freezing ResNet-18 Params
for param in embedding_resnet.base_model.parameters():
    param.requires_grad = False

# Defining loss function and optimizer for the classifier
criterion_classifier = nn.CrossEntropyLoss()
optimizer_classifier = optim.Adam(embedding_resnet.fc.parameters(), lr=0.001)

# Training with classifier
loss1 = []
accuracy = []
num_epochs_classifier = 10
for epoch in range(num_epochs_classifier):
    embedding_resnet.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader_train:
        images, labels = images.to(device), labels.to(device)
        optimizer_classifier.zero_grad()

        outputs = embedding_resnet(images)
        loss = criterion_classifier(outputs, labels)

        loss.backward()
        optimizer_classifier.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(loader_train)
    acc = 100 * correct / total
    print(f"Epoch: {epoch}, Loss: {avg_loss}, Accuracy: {acc}%")
    loss1.append(avg_loss)
    accuracy.append(acc)

sns.lineplot(x=range(len(loss1)), y=loss1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Classifier Loss Change in Every Epoch")
plt.show()

sns.lineplot(x=range(len(accuracy)), y=accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Classifier Accuracy Change in Every Epoch")
plt.show()


