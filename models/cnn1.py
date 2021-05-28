import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNN1(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.fc1 = nn.Linear(16 * 7 *7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pooling(x)
        x = F.relu(self.conv2(x))
        x = self.pooling(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x