from pytorch_lightning.core.lightning import LightningModule
import torch
from torch import nn
from torch.nn import functional as F


class CNN(LightningModule):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 16, 4)
        self.conv2 = nn.Conv2d(16, 16, 4)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)

        self.dropout = nn.Dropout(0.25)

        # self.fc1 = nn.LazyLinear(128)
        self.fc1 = nn.Linear(32 * 6 * 14, 120)

    def forward(self, x):
        batch_size, height, width = x.size()
        x = x.view(batch_size, 1, height, width)  # convert to one channel

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x)))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return x
