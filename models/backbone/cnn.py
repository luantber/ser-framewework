from pytorch_lightning.core.lightning import LightningModule
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

class CNN(LightningModule):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 32, 4, padding="same")
        
        self.conv2 = nn.Conv2d(32, 32, 4, padding="same")  # 32xmxn
        self.conv3 = nn.Conv2d(32, 32, 3, padding="same")  # 32xmxn


        self.conv4 = nn.Conv2d(32, 32, 3,padding="same")
        self.conv5 = nn.Conv2d(32, 32, 3,padding="same")

        # self.conv6 = nn.Conv2d(32, 32, 3)
        # self.conv7 = nn.Conv2d(32, 32, 3)

        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.LazyLinear(128)
        # self.fc1 = nn.Linear(32 * 6 * 14, 120)

    def forward(self, x):
        batch_size, height, width = x.size()
        x = x.view(batch_size, 1, height, width)  # convert to one channel

        x = self.pool(F.relu(self.conv1(x)))

        x = F.relu(self.conv2(x))
        res = x
        x = F.relu(self.conv3(x))
        # print("res", res.shape, "x", x.shape)
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x + res)))
        x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))
        # x = F.relu(self.conv7(x))



        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return x
