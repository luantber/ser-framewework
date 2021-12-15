import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
import torchmetrics

from models.backbone.cnn import CNN
from pytorch_lightning.core.lightning import LightningModule


class CNNDim(LightningModule):
    def __init__(self, lr, loss="mse"):
        super().__init__()

        # Atributos Clase
        self.lr = lr
        self.n_dimensions = 3

        # Configure Network Architecture
        self.cnn = CNN()
        self.fc = nn.Linear(128, self.n_dimensions)

        # Configure Loss and Metric of Precision
        if loss == "mse":
            self.loss = F.mse_loss
        elif loss == "ccc":
            raise NotImplementedError("CCC Loss not implemented yet")
        else:
            raise ValueError("Loss not supported")

        self.metric = torchmetrics.MeanAbsoluteError

        # BoilerpLate Dimensions
        self.accuracy = self.metric()
        self.accuracy_val = self.metric()

        self.d1 = self.metric()
        self.d2 = self.metric()
        self.d3 = self.metric()

        self.d1_val = self.metric()
        self.d2_val = self.metric()
        self.d3_val = self.metric()

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

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
        loss = self.loss(logits, y)

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

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=(self.lr))
