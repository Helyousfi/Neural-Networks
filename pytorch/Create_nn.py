# Imports
from pickletools import optimize
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

# Create a fully connected network
class neural_network(nn.Module):
    def __init__(self, input_size, num_classes):
        super(neural_network, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        #self.fc0 = nn.Flatten()
        self.fc1 = nn.Linear(self.input_size, 50)
        self.fc2 = nn.Linear(50, self.num_classes)

    def forward(self, x):
        #x = self.fc0(x)
        # F.relu nn.ReLU
        x = F.relu(self.fc1(x))
        return F.sigmoid(self.fc2(x))

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 28 * 28
batch_size = 64
learning_rate = 0.001
num_classes = 10
num_epochs = 20

# Data-augmentation
data_transforms = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize(size=(28,28))]
)

# Load dataset
train_data = datasets.MNIST(root='dataset/', download=True, train=True, transform=data_transforms)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = datasets.MNIST(root='dataset/', download=False, train=False, transform=transforms.ToTensor())
test_dataloader = DataLoader(dataset=test_data, batch_size=2, shuffle=False)



# Define the model
model = neural_network(input_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking the accuracy on training dataset...")
    else:
        print("Checking the accuracy on validation dataset...")
    
    num_corrects = 0.0
    num_samples = 0.0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x.to(device)
            y.to(device)
            x = x.reshape(x.shape[0], -1)
            predictions = model(x)
            _, predictions = predictions.max(1)
            num_corrects += (predictions == y).sum()
            num_samples += batch_size
        print(f"Got the following accuracy : {num_corrects / num_samples}")
    model.train()


for epoch in range(num_epochs):
    for idx, (image, target) in enumerate(train_dataloader):
        # Move everything to device
        image = image.to(device)
        target = target.to(device)
        # Reshape
        image = image.reshape(image.shape[0], -1) # (64, 28*28)
        # predict 
        prediction = model(image) 
        # compute the loss
        loss = criterion(prediction, target)
        # update the parameters 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        check_accuracy(test_dataloader, model)





if __name__ == "__main__":
    if 0:
        model = neural_network(input_size, num_classes)
        image = torch.randn(2, 28 * 28)
        print(image.shape)
        print(model(image))