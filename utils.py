import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import wandb

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data
def create_dataloaders(config):
    train_data = datasets.MNIST(root = 'data', train = True, transform = ToTensor(), download = True)
    test_data = datasets.MNIST(root = 'data', train = False, transform = ToTensor(),download = True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'], shuffle=True, num_workers=1)
    return train_loader, test_loader

# CNN network
def cnn(config):
    net = nn.Sequential()
    net.add_module("conv1",nn.Conv2d(in_channels=1,out_channels=config['hidden_layer_width'], kernel_size=3))
    net.add_module("pool1",nn.MaxPool2d(kernel_size=2,stride=2)) 
    net.add_module("conv2",nn.Conv2d(in_channels=config['hidden_layer_width'],
                                     out_channels=config['hidden_layer_width'],kernel_size=5))
    net.add_module("pool2",nn.MaxPool2d(kernel_size=2,stride=2))
    net.add_module("dropout",nn.Dropout2d(p=config['dropout_rate']))
    net.add_module("adaptive_pool",nn.AdaptiveMaxPool2d((1,1)))
    net.add_module("flatten",nn.Flatten())
    net.add_module("linear1",nn.Linear(config['hidden_layer_width'], config['hidden_layer_width']))
    net.add_module("relu",nn.ReLU())
    net.add_module("linear2",nn.Linear(config['hidden_layer_width'], 10))
    net.to(device)
    return net 

def train_epoch(model, train_loader, optimizer):
    model.train()
    for batch in train_loader:
        features, labels = batch
        features, labels = features.to(device), labels.to(device)

        preds = model(features)
        loss = nn.CrossEntropyLoss()(preds, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    return model, loss

def eval_epoch(model, test_loader):
    model.eval()
    accurate = 0
    num_elems = 0
    for batch in test_loader:
        features, labels = batch
        features, labels = features.to(device),labels.to(device)
        with torch.no_grad():
            preds = model(features)
            val_loss = nn.CrossEntropyLoss()(preds, labels)
        predictions = preds.argmax(dim=-1)
        accurate_preds =  (predictions==labels)
        num_elems += accurate_preds.shape[0]
        accurate += accurate_preds.long().sum()

    val_acc = accurate.item() / num_elems
    return val_acc, val_loss

def show_cases(model, show_num):
    cases = []
    test_data = datasets.MNIST(root = 'data', train = False, transform = ToTensor(),download = True)
    for i in range(show_num):
        features, label = test_data[i]
        tensor = features.to(device)
        pred = torch.argmax(model(tensor[None,...])).cpu().item()
        image = wandb.Image(features.permute(1,2,0).numpy())
        cases.append([image, label, pred])
    return cases