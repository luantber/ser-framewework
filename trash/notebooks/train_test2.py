import sys
sys.path.append('..')

import torch
from datasets.ravdess import Ravdess
from datasets.utils import randomcrop, centercrop
from models.cnn import CNN
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

dataset = Ravdess(
    "../ser_datasets/ravdess/train.csv",
    "../ser_datasets/ravdess/audios",
)


int_train = int(0.7*len(dataset))

train, test = random_split(dataset, [ int_train , len(dataset) - int_train ] ,  generator=torch.Generator().manual_seed(42) )

# print(train)
train.dataset.set_transform([randomcrop])
test.dataset.set_transform([centercrop])

train_dataloader = DataLoader( train ,
    batch_size=128 , shuffle=True, num_workers = 3 , persistent_workers=True, 

)
test_dataloader = DataLoader( test ,
    batch_size=128 , shuffle=False, num_workers = 3 , persistent_workers=True, 
)



## Training 
wandb_logger = WandbLogger(project="ser", config= dict(
    batch_size = 128,
    max_epochs = 250,
    architecture= "CNNv1",
) )

net = CNN(0.001)
trainer = Trainer(gpus=1,logger=wandb_logger,max_epochs=250, auto_lr_find=True,precision=16)
trainer.fit(net,train_dataloader,test_dataloader)