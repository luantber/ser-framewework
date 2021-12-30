# resnet18 = models.resnet18(pretrained=True)
import torchvision.models as models
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
import torchmetrics
from models.metrics import MSE, CCC, mse_loss_custom, ccc_loss_custom, ccc

from models.backbone.cnn import CNN
from pytorch_lightning.core.lightning import LightningModule


class Resnet(LightningModule):
    def __init__(self, lr, loss="mse"):
        super().__init__()

        # Atributos Clase
        self.lr = lr
        self.n_dimensions = 3

        # Configure Network Architecture
        self.cnn = models.resnet18(pretrained=True)
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.cnn.fc = nn.Sequential(
            nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(self.n_dimensions)
        )

        # Configure Loss and Metric of Precision
        if loss == "mse":
            # self.loss = F.mse_loss
            self.loss = mse_loss_custom
        elif loss == "ccc":
            self.loss = ccc_loss_custom
        else:
            raise ValueError("Loss not supported")

        # BoilerpLate Dimensions

        self.ccc_train_avg = CCC()
        self.ccc_train_v = CCC()
        self.ccc_train_a = CCC()
        self.ccc_train_d = CCC()

        self.ccc_val_avg = CCC()
        self.ccc_val_v = CCC()
        self.ccc_val_a = CCC()
        self.ccc_val_d = CCC()

    def forward(self, x):
        batch_size, height, width = x.size()
        x = x.view(batch_size, 1, height, width)
        x = torch.cat((x, x, x), 1)

        x = self.cnn(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        ccc_v = ccc(logits[:, 0], y[:, 0])
        ccc_a = ccc(logits[:, 1], y[:, 1])
        ccc_d = ccc(logits[:, 2], y[:, 2])

        self.ccc_train_v(ccc_v)
        self.ccc_train_a(ccc_a)
        self.ccc_train_d(ccc_d)

        a = 0.1
        b = 0.5
        self.ccc_train_avg(a * ccc_v + b * ccc_a + (1 - a - b) * ccc_d)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_epoch_end(self, outs):

        self.log("train/ccc_v", self.ccc_train_v)
        self.log("train/ccc_a", self.ccc_train_a)
        self.log("train/ccc_d", self.ccc_train_d)
        self.log("train/ccc_avg", self.ccc_train_avg)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        ccc_v = ccc(logits[:, 0], y[:, 0])
        ccc_a = ccc(logits[:, 1], y[:, 1])
        ccc_d = ccc(logits[:, 2], y[:, 2])

        self.ccc_val_v(ccc_v)
        self.ccc_val_a(ccc_a)
        self.ccc_val_d(ccc_d)

        a = 0.1
        b = 0.5
        self.ccc_val_avg(a * ccc_v + b * ccc_a + (1 - a - b) * ccc_d)

        self.log("val/loss", loss, prog_bar=True)

    def validation_epoch_end(self, outs):

        self.log("val/ccc_v", self.ccc_val_v)
        self.log("val/ccc_a", self.ccc_val_a)
        self.log("val/ccc_d", self.ccc_val_d)
        self.log("val/ccc_avg", self.ccc_val_avg)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=(self.lr))
