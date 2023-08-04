# Libraries
import torch
import torch.nn as NN
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets

# Hyperparameters
image_size = 256
batch_size = 32
learning_rate = 0.001
num_epochs = 25

# Importing Data And Applying Data Augmentation
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root="Tomato", transform=transform)


# Splitting Data into train and test data
def train_test_ds(data, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(data))), test_size=val_split)
    train_data = Subset(data, train_idx)
    val_data = Subset(data, val_idx)

    return train_data, val_data


train_data, val_data = train_test_ds(dataset)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Defining CNN
class CNN(NN.Module):
    def __init__(self, in_channel=3, num_classes=15):
        super(CNN, self).__init__()
        self.conv1 = NN.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.pool = NN.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = NN.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv3 = NN.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.conv4 = NN.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.conv5 = NN.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.conv6 = NN.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.fc1 = NN.Linear(256, 64)
        self.fc2 = NN.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


model = CNN().to(device)

# Loss Function and Optimizer
criterion = NN.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Finding Accuracy on Validation Data
def check_accuracy(loader, tmodel):
    num_correct = 0
    num_samples = 0
    tmodel.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = tmodel(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    tmodel.train()

    return float(num_correct) / float(num_samples) * 100


# Training Model

best_acc = 0
patience = 5
counter = 0

for epoch in range(num_epochs):
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)

        scores = model(data)
        loss = criterion(scores, label)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    val_acc = check_accuracy(val_loader, model)

    print("Epoch_No:-", epoch, " Validation Accuracy:-", val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        counter = 0
        torch.save(model, 'Models/BestModel.pth.tar')
    else:
        counter += 1

    if counter >= patience:
        print("Stopping Training.")
        break

