from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
import numpy as np
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Initial layers - reduce channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=1),  # Increased from 8
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05)  # Reduced dropout
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, 3, padding=1),  # Increased from 16
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.05)
        )
        
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Middle layers
        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 20, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.05)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(20, 20, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.05)
        )
        
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Final layers
        self.conv5 = nn.Sequential(
            nn.Conv2d(20, 20, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.05)
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv6 = nn.Conv2d(20, 10, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.gap(x)
        x = self.conv6(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Calculate training accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        train_loss += loss.item()
        
        # Update progress bar
        pbar.set_description(desc=f'Epoch: {epoch} | Train Loss: {loss.item():.4f} | Batch: {batch_idx}')
    
    # Print epoch summary
    train_loss /= len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'\nEpoch: {epoch}')
    print(f'Training Set - Average Loss: {train_loss:.4f}, Accuracy: {correct}/{total} ({train_accuracy:.2f}%)')
    
    return train_accuracy  # Return the training accuracy

def test(model, device, test_loader, set_name="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= total
    accuracy = 100. * correct / total

    print(f'{set_name} Set - Average Loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    
    return accuracy

def get_train_test_loader(batch_size=1024, test_size=0.167, shuffle=True):
    """
    Split training data into 50k training and 10k test
    test_size = 0.167 because 10k/60k = 0.167
    """
    dataset = datasets.MNIST(
        '../data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(test_size * num_train))

    if shuffle:
        np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=1, pin_memory=True
    )

    test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler,
        num_workers=1, pin_memory=True
    )

    return train_loader, test_loader

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    torch.manual_seed(1)
    batch_size = 1024

    # Get train and validation loaders (50k/10k split)
    train_loader, valid_loader = get_train_test_loader(batch_size=batch_size)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.004)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5,
        patience=2,
        verbose=True
    )

    best_accuracy = 0
    for epoch in range(1, 19):
        # Training
        train_accuracy = train(model, device, train_loader, optimizer, epoch)
        
        # Validation after each epoch
        print("\nValidation Results:")
        valid_accuracy = test(model, device, valid_loader, set_name="Validation")
        scheduler.step(valid_accuracy)
        
        # Save best model based on validation accuracy
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(model.state_dict(), 'mnist_model.pth')
            print(f"New best model saved! Accuracy: {valid_accuracy:.2f}%")
        
        if valid_accuracy >= 99.4:
            print(f"Target accuracy achieved! Final accuracy: {valid_accuracy:.2f}%")
            break

    # Load best model for final test
    model.load_state_dict(torch.load('mnist_model.pth'))
    print("\nFinal Test Results:")
    final_accuracy = test(model, device, valid_loader, set_name="Test")
    print(f"Best accuracy achieved: {best_accuracy:.2f}%")

if __name__ == '__main__':
    main() 