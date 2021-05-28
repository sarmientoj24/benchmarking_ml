import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
import wandb
from models import CNN2 as CNN
from utils import DatasetLoader
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

    # params and Hyper params
    input_size  = 224 * 224 *3   
    output_size = 2
    num_workers = 1
    batch_size = 128
    epochs = args.epochs
    learning_rate = 0.01
    n_features = 2 # hyperparameter

    # define training and test data directories
    data_dir = 'dataset/CATDOG/'
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # Create transformers
    image_size = (224, 224)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
                                    transforms.Resize(image_size), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean, std)])
    test_transforms = transforms.Compose([
                                    transforms.Resize(image_size), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean, std)])

    train_dataset = DatasetLoader(train_dir, transform=train_transform)
    test_dataset = DatasetLoader(test_dir, transform=test_transforms)

    ## Load data
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, 
        num_workers=num_workers)


    # Model
    model = CNN(input_size, n_features, output_size)

    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Init Wandb connection
    wandb.init(
        project=args.proj, 
        entity=args.entity, 
        name=args.name
    )

    start_time = time.time()
    for _ in range(epochs):
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
        accuracy = check_accuracy(test_loader, model, device)
        print("Accuracy: ", accuracy)
        wandb.log({"accuracy": accuracy})
    
    print(f"Train-Val duration: {time.time() - start_time}s")

    # Check accuracy
    time_s = time.time()
    print(f"Accuracy on training set: {check_accuracy(train_loader, model, device)*100:.2f}")
    print(f"Training Validation duration: {time.time() - time_s}s")

    time_s = time.time()
    print(f"Accuracy on test set: {check_accuracy(test_loader, model, device)*100:.2f}")
    print(f"Training Validation duration: {time.time() - start_time}s")


if __name__ == '__main__':
    main()