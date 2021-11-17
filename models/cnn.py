import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule


from torch.optim import Adam
import torchmetrics

class CNN(LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)


        self.fc1 = nn.Linear(32 * 6 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)

        self.accuracy = torchmetrics.Accuracy()
        self.accuracy_val = torchmetrics.Accuracy()
        self.confusion = torchmetrics.ConfusionMatrix(num_classes=8)

    def forward(self, x):
        batch_size,  height, width = x.size()

        x = x.view(batch_size, 1 , height , width)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print("sahpe1",x.shape)
        x = self.pool(F.relu(self.conv3(x)))

        # print("sahpe",x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.accuracy(logits, y)
        self.log('train_acc_step', self.accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.accuracy_val(logits, y)

        

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy)
    
    def validation_epoch_end(self, outs):
        self.log('train_val_epoch', self.accuracy_val)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=(self.lr))

