import torch
from torch.utils.data.dataset import Subset
from datasets.ravdess import Ravdess
from datasets.utils import randomcrop, centercrop
from models.cnn import CNN
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
# import sys
# sys.path.append('..')

def run( model , config ):

    dataset_train = Ravdess(
        "ser_datasets/ravdess/train.csv",
        "ser_datasets/ravdess/audios",
        transform=[randomcrop]
    )

    dataset_test = Ravdess(
        "ser_datasets/ravdess/train.csv",
        "ser_datasets/ravdess/audios",
        transform=[centercrop]
    )

    indexes =  np.arange( len(dataset_train) )
    np.random.shuffle( indexes )
    split = int( len(indexes) * 0.7 )
    train, test = indexes[:split] , indexes[split:]

    assert( len(train) + len(test) == len(indexes) )


    train_subset = Subset(dataset_train, train)
    test_subset =  Subset(dataset_test, test)
    

    train_dataloader = DataLoader( train_subset ,
        batch_size=config["batch_size"] , shuffle=True, num_workers = 3 , persistent_workers=True, 

    )
    test_dataloader = DataLoader( test_subset ,
        batch_size=config["batch_size"] , shuffle=False, num_workers = 3 , persistent_workers=True, 
    )

    
    ## Training 
    wandb_logger = WandbLogger(project="ser_ravdess", config=config )

    net = model(config["lr"])
    trainer = Trainer(gpus=1,logger=wandb_logger,max_epochs=config["epochs"],precision=16)
    trainer.fit(net,train_dataloader,test_dataloader)

    wandb.finish()