from datasets.kusisqadim import KusisqaDim
from datasets.utils import randomcrop, centercrop

from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader


from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb
import numpy as np

def run( model , config ):
    dataset_train = KusisqaDim(
        "ser_datasets/kusisqadim/train.csv",
        "ser_datasets/kusisqadim/audios",
        transform=[randomcrop]
    )

    dataset_test = KusisqaDim(
        "ser_datasets/kusisqadim/train.csv",
        "ser_datasets/kusisqadim/audios",
        transform=[centercrop]
    )

    indexes =  np.arange( len(dataset_train) )
    np.random.shuffle( indexes )
    split = int( len(indexes) * 0.8 )
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
    wandb_logger = WandbLogger(project="ser_kusisqaDim", config=config , save_dir="logs")

    net = model(config["lr"],config["out"])
    trainer = Trainer(gpus=1,logger=wandb_logger,max_epochs=config["epochs"],precision=16)
    trainer.fit(net,train_dataloader,test_dataloader)

    wandb.finish()

def get_number_classes():
    dataset = KusisqaDim(
        "ser_datasets/iemocap/train.csv",
        "ser_datasets/iemocap/audios",
        transform=[randomcrop]
    )
    return dataset.number_classes
