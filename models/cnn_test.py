import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule


from torch.optim import Adam
import torchmetrics


class CNNDim(LightningModule):
    def __init__(self, lr, out):
        super().__init__()
        self.lr = lr

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 16, 4)
        self.conv2 = nn.Conv2d(16, 16, 4)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)

        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(32 * 1 * 5, 128)
        self.fc3 = nn.Linear(128, 3)

        self.accuracy = torchmetrics.MeanAbsoluteError()
        self.accuracy_val = torchmetrics.MeanAbsoluteError()

        self.d1 = torchmetrics.MeanAbsoluteError()
        self.d2 = torchmetrics.MeanAbsoluteError()
        self.d3 = torchmetrics.MeanAbsoluteError()

        self.d1_val = torchmetrics.MeanAbsoluteError()
        self.d2_val = torchmetrics.MeanAbsoluteError()
        self.d3_val = torchmetrics.MeanAbsoluteError()

        ## Results
        self.accuracy_test = torchmetrics.MeanAbsoluteError()

    def forward(self, x):
        batch_size, height, width = x.size()

        x = x.view(batch_size, 1, height, width)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x)))

        # print(x.shape,">>>>>>>>>>>>>")
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.mse_loss(logits, y)

        self.accuracy(logits, y)

        self.d1(logits[:, 0], y[:, 0])
        self.d2(logits[:, 1], y[:, 1])
        self.d3(logits[:, 2], y[:, 2])

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def training_epoch_end(self, outs):
        self.log("train/acc", self.accuracy)
        self.log("train/d1", self.d1)
        self.log("train/d2", self.d2)
        self.log("train/d3", self.d3)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.mse_loss(logits, y)

        self.accuracy_val(logits, y)
        self.d1_val(logits[:, 0], y[:, 0])
        self.d2_val(logits[:, 1], y[:, 1])
        self.d3_val(logits[:, 2], y[:, 2])

        self.log("val/loss", loss, prog_bar=True)

    def validation_epoch_end(self, outs):
        self.log("val/acc", self.accuracy_val)
        self.log("val/d1", self.d1_val)
        self.log("val/d2", self.d2_val)
        self.log("val/d3", self.d3_val)

    # def test_step(self,batch,idx):
    #     x, y = batch
    #     logits = self(x)
    #     self.accuracy_test(logits, y)

    # def test_epoch_end(self,out):
    #     self.log("test/acc",self.accuracy_test)
    #     self.log("test/f1",self.f1_test)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=(self.lr))
