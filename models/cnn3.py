import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule


from torch.optim import Adam
import torchmetrics

class CNN3(LightningModule):
    def __init__(self, lr, out):
        super().__init__()
        self.lr = lr

        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)

        self.fc1 = nn.Linear(64 * 2 * 6, 1024)
        self.fc3 = nn.Linear(1024, out)

        self.accuracy = torchmetrics.Accuracy()
        self.accuracy_val = torchmetrics.Accuracy()

        ## Results
        self.accuracy_test = torchmetrics.Accuracy()
        self.f1_test = torchmetrics.F1()
        
        self.confusion = torchmetrics.ConfusionMatrix(num_classes=8)

    def forward(self, x):
        batch_size,  height, width = x.size()

        x = x.view(batch_size, 1 , height , width)

        

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # print("sahpe1",x.shape)
        

        # print("sahpe",x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.accuracy(logits, y)
        self.log('train/acc_step', self.accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.accuracy_val(logits, y)


    def test_step(self,batch,idx):
        x, y = batch
        logits = self(x)
        self.accuracy_test(logits, y)
        self.f1_test(logits, y)
    
    def test_epoch_end(self,out):
        self.log("test/acc",self.accuracy_test)
        self.log("test/f1",self.f1_test)

    
    def training_epoch_end(self, outs):
        self.log('train/acc', self.accuracy)
    
    def validation_epoch_end(self, outs):
        self.log('val/acc', self.accuracy_val)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=(self.lr))

