import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
import torchmetrics
from models.metrics import MSE, CCC, mse_loss_custom, ccc_loss_custom

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
            # self.loss = F.mse_loss
            self.loss = mse_loss_custom
        elif loss == "ccc":
            self.loss = ccc_loss_custom
        else:
            raise ValueError("Loss not supported")

        # BoilerpLate Dimensions
        self.mse_train = MSE()
        self.mse_val = MSE()

        self.ccc_train = CCC()
        self.ccc_val = CCC()

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        self.mse_train(logits, y)
        self.ccc_train(logits, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_epoch_end(self, outs):

        self.log("train/mse", self.mse_train)
        self.log("train/ccc", self.ccc_train)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        self.mse_val(logits, y)
        self.ccc_val(logits, y)

        self.log("val/loss", loss, prog_bar=True)

    def validation_epoch_end(self, outs):
        self.log("val/mse", self.mse_val)
        self.log("val/ccc", self.ccc_val)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=(self.lr))
