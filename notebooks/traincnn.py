import sys
sys.path.append('..')


from datasets.ravdess import Ravdess
from datasets import utils
from models.cnn import CNN
from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
import torchmetrics
import pandas as pd 
import wandb
from multiprocessing import Manager


wandb_logger = WandbLogger(project="ser", config= dict(
    batch_size = 256,
    max_epochs = 180,
    architecture= "CNN"
) )

manager = Manager()
shared_dict1 = manager.dict()
shared_dict2 = manager.dict()
data = Ravdess(
    "../ser_datasets/ravdess/train.csv",
    "../ser_datasets/ravdess/audios",
    shared_dict1,
    shared_dict2,
    transform = [utils.randomcrop]
    )


dataloader = DataLoader( data ,
    batch_size=256 , shuffle=True, num_workers=5,pin_memory=True
)

net = CNN()

trainer = Trainer(gpus=1,logger=wandb_logger,max_epochs=180,profiler="simple")
trainer.fit(net,dataloader)


for x,y in dataloader:
    predict = net(x)
    net.confusion(predict,y)

ct = pd.DataFrame(net.confusion.compute(), columns=[i for i in range(8)])
table = wandb.Table(dataframe=ct)
wandb.log({"c_m":table})




