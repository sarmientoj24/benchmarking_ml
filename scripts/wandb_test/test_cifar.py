import torch
import torchvision
import torchvision.models as models
import torch.nn.functional as F  
import torchvision.datasets as dataset 
import torchvision.transforms as transforms  
from torch import optim  
from torch import nn  
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import  check_accuracy
import wandb
import argparse
import time
import numpy as np
from timerit import Timerit

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj", help="wandb project",
                        type=str)
    parser.add_argument("--name", help="wandb experiment name",
                        type=str)
    parser.add_argument("--entity", help="wandb entity name",
                        type=str)
    parser.add_argument("--epochs", help="epochs",
                        type=int)

    args = parser.parse_args()

    # Initialize torch seed
    torch.random.manual_seed(42)

    # initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Hyperparameters
    num_classes = 10
    learning_rate = 0.001
    batch_size = 128
    epochs = args.epochs

    # Create Transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # Load Data
    train_dataset = dataset.CIFAR10(
        root="dataset/CIFAR10/", 
        train=True, transform=transform_train, 
        download=True)
    test_dataset = dataset.CIFAR10(
        root="dataset/CIFAR10/", 
        train=False, transform=transform_test, 
        download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize network
    model = models.vgg11(pretrained=True)

    params = [
#        'classifier.0.weight', 
#        'classifier.0.bias', 
        'classifier.3.weight', 
        'classifier.3.bias', 
        'classifier.6.weight', 
        'classifier.6.bias'
    ]

    model.classifier[6] = nn.Linear(4096, num_classes)
    for name, param in model.named_parameters():
        if name in params:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Init Wandb connection
    wandb.init(
        project=args.proj, 
        entity=args.entity, 
        name=args.name
    )

    # Train
    start_time = time.time()
    per_epoch_train_duration = []
    per_epoch_with_val_duration = []
    for _ in range(epochs):
        epoch_time = time.time()
        model.train()
        for _, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device=device)
            targets = targets.to(device=device)
            
            # forward
            output = model(data)
            loss = criterion(output, targets)
            wandb.log({"loss": loss})

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
        per_epoch_train_duration.append(time.time() - epoch_time)
        accuracy = check_accuracy(test_loader, model, device)
        per_epoch_with_val_duration.append(time.time() - epoch_time)
        print("Accuracy: ", accuracy)
        wandb.log({"accuracy": accuracy})
    
        print("Epoch Train Duration Mean: ", np.mean(per_epoch_train_duration))
    print("Epoch Train-Val Duration Mean: ", np.mean(per_epoch_with_val_duration))
    print("Epoch Train Duration STD: ", np.std(per_epoch_train_duration))
    print("Epoch Train-Val Duration STD: ", np.std(per_epoch_with_val_duration))
    print("Full Train-Only Duration: ", np.sum(per_epoch_train_duration))

    print("Training duration: ", time.time() - start_time)
    # Check accuracy
        # Check accuracy
    print(f"Accuracy on training set: {check_accuracy(train_loader, model, device)*100:.2f}")
    for _ in Timerit(num=5, verbose=2):
        check_accuracy(train_loader, model, device)

    # Validation Time using Test Set
    print(f"Accuracy on test set: {check_accuracy(test_loader, model, device)*100:.2f}")
    for _ in Timerit(num=5, verbose=2):
        check_accuracy(test_loader, model, device)


if __name__ == '__main__':
    main()
