import multiprocessing as mp
import psutil
import numpy as np

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
import numpy as np
from timerit import Timerit


def run_model():
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

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
        per_epoch_train_duration.append(time.time() - epoch_time)
        accuracy = check_accuracy(test_loader, model, device)
        per_epoch_with_val_duration.append(time.time() - epoch_time)

    print("Epoch Train Duration Mean: ", np.mean(per_epoch_train_duration))
    print("Epoch Train-Val Duration Mean: ", np.mean(per_epoch_with_val_duration))
    print("Epoch Train Duration STD: ", np.std(per_epoch_train_duration))
    print("Epoch Train-Val Duration STD: ", np.std(per_epoch_with_val_duration))
    print("Full Train-Only Duration: ", np.sum(per_epoch_train_duration))

    print("Training duration: ", time.time() - start_time)

def check_train_accuracy():
    for _ in Timerit(num=5, verbose=2):
        check_accuracy(train_loader, model, device)


def check_test_accuracy():
    # Validation Time using Test Set
    for _ in Timerit(num=5, verbose=2):
        check_accuracy(test_loader, model, device)

import pandas as pd



def check_usage(system_info, p):
    system_info['cpu_percent'].append(p.cpu_percent())
    system_info['ram_percent'].append(p.memory_percent())
    system_info['cpu_threads'].append(p.num_threads())
    system_info['memory_rss'].append(p.memory_full_info().rss / 1024 / 1024)
    system_info['memory_vms'].append(p.memory_full_info().vms / 1024 / 1024)
    system_info['memory_shared'].append(p.memory_full_info().shared / 1024 / 1024)
    system_info['memory_data'].append(p.memory_full_info().data / 1024 / 1024)
    system_info['memory_swap'].append(p.memory_full_info().swap / 1024 / 1024)
    system_info['memory_uss'].append(p.memory_full_info().uss / 1024 / 1024)
    system_info['memory_pss'].append(p.memory_full_info().pss / 1024 / 1024)
    system_info['swap_memory_used'].append(p.memory_full_info().swap / 1024 / 1024)

def monitor(target):
    worker_process = mp.Process(target=target)
    worker_process.start()
    p = psutil.Process(worker_process.pid)

    system_info = {
        'cpu_percent': [],
        'ram_percent': [],
        'cpu_threads': [],
        'memory_rss': [],
        'memory_vms': [],
        'memory_shared': [],
        'memory_data': [],
        'memory_swap': [],
        'memory_uss': [],
        'memory_pss': [],
        'swap_memory_used': []
    }
    # log cpu usage of `worker_process` every 10 ms
    while worker_process.is_alive():
        time.sleep(sleep_interval)
        check_usage(system_info, p)
    worker_process.join()
    return system_info

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", help="epochs", default=10,
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

####### Log
folder = 'tmp'
dataset = 'catdog'
mode = 'train'
sleep_interval = 0.5
training_val = monitor(target=run_model)
pd.DataFrame(training_val).to_csv(f"./{folder}/{dataset}_{mode}.csv", index=False)

# mode = 'train_acc'
# sleep_interval = 0.5
# train_accuracy = monitor(target=check_train_accuracy)
# pd.DataFrame(train_accuracy).to_csv(f"./{folder}/{dataset}_{mode}.csv", index=False)

mode = 'test_acc'
sleep_interval = 0.5
test_accuracy = monitor(target=check_test_accuracy)
pd.DataFrame(test_accuracy).to_csv(f"./{folder}/{dataset}_{mode}.csv", index=False)
