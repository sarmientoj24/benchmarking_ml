import torch
import torchvision 
import torch.nn.functional as F  
import torchvision.datasets as dataset 
import torchvision.transforms as transforms  
from torch import optim  
from torch import nn  
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from models import CNN1 as CNN
from utils import  check_accuracy
import argparse
import time


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
    in_channels = 1
    num_classes = 10
    learning_rate = 0.001
    batch_size = 64
    num_epochs = args.epochs

    # Load Data
    train_dataset = dataset.MNIST(
        root="dataset/MNIST/", 
        train=True, transform=transforms.ToTensor(), 
        download=True)
    test_dataset = dataset.MNIST(
        root="dataset/MNIST/", 
        train=False, transform=transforms.ToTensor(), 
        download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Initialize network
    model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

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
    for _ in range(num_epochs):
        model.train()
        for _, (data, targets) in enumerate(tqdm(train_loader)):
            # get data to cuda
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
            
        accuracy = check_accuracy(test_loader, model, device)
        wandb.log({"accuracy": accuracy})

    print("Training duration: ", time.time() - start_time)
    # Check accuracy
    time_s = time.time()
    print(f"Accuracy on training set: {check_accuracy(train_loader, model, device)*100:.2f}")
    print(f"Training Validation duration: {time.time() - time_s}s")

    time_s = time.time()
    print(f"Accuracy on test set: {check_accuracy(test_loader, model, device)*100:.2f}")
    print(f"Training Validation duration: {time.time() - start_time}s")


if __name__ == '__main__':
    main()