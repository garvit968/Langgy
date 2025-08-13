import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create a fully connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# model = NN(784, 10)
# x = torch.randn(64, 784)
# print(model(x).shape)

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# lOAD Data
train_dataset = datasets.MNIST(root='dataset/', train= True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train= False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialise Network
model = NN(input_size=input_size, num_classes=num_classes).to(device=device)

# Loss Function
criterion = nn.CrossEntropyLoss()
optmizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train the Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to Cuda 
        data = data.to(device = device)
        targets = targets.to(device = device)

        # Get data to correct
        data = data.reshape(data.shape[0], -1)
        
        # forward
        score = model(data)
        loss = criterion(score, targets)

        # backword
        optmizer.zero_grad()
        loss.backward()

        # gradient or adam step
        optmizer.step()

# Check Accuracy
def check_accuracy(loader, model):
    num_corr = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x =x.to(device=device)
            y = y.to(device = device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_corr+=(predictions==y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_corr}/{num_samples} with accuracy of {float(num_corr)/float(num_samples)*100:.2f}')
    model.train()
    
check_accuracy(train_loader, model=model)
check_accuracy(test_loader, model=model)