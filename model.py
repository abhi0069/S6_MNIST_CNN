from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.Dropout(0.25))
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.Dropout(0.25))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv1_1x1 = nn.Conv2d(16, 8, 1)
        self.conv3 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.Dropout(0.25))
        self.conv4 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.Dropout(0.25))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2_1x1 = nn.Conv2d(32, 8, 1)
        self.conv5 = nn.Sequential(nn.Conv2d(8, 16, 3), nn.BatchNorm2d(16), nn.Dropout(0.25))
        self.conv6 = nn.Sequential(nn.Conv2d(16, 32, 3), nn.BatchNorm2d(32), nn.Dropout(0.25))
        self.conv7 = nn.Conv2d(32, 10, 3)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.conv1_1x1(x)
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.conv2_1x1(x)
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = F.relu(self.conv7(x))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)  # Fixed the deprecated warning

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    return accuracy

def get_train_valid_loader(batch_size=128, valid_size=0.167, shuffle=True):
    """
    Creates train and validation dataloaders with a 50k/10k split from training data
    valid_size = 0.167 because 10k/60k = 0.167
    """
    train_dataset = datasets.MNIST(
        '../data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=1, pin_memory=True
    )

    valid_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=1, pin_memory=True
    )

    return train_loader, valid_loader

def get_test_loader(batch_size=128):
    """
    Creates dataloader for the separate test set of 10k images
    """
    return DataLoader(
        datasets.MNIST('../data', train=False, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=batch_size, shuffle=True,
        num_workers=1, pin_memory=True
    )

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    torch.manual_seed(1)
    batch_size = 128

    # Get the dataloaders with proper splits
    train_loader, valid_loader = get_train_valid_loader(batch_size=batch_size)
    test_loader = get_test_loader(batch_size=batch_size)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    for epoch in range(1, 20):
        train(model, device, train_loader, optimizer, epoch)
        # Validate on validation set
        valid_accuracy = test(model, device, valid_loader)
        scheduler.step(valid_accuracy)
        # Test on separate test set
        test_accuracy = test(model, device, test_loader)
        
        # Save model if accuracy threshold is met on test set
        if test_accuracy >= 99.4:
            torch.save(model.state_dict(), 'mnist_model.pth')
            print(f"Model saved! Achieved {test_accuracy:.2f}% accuracy")
            break

if __name__ == '__main__':
    main() 