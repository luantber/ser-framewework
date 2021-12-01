import sys
sys.path.append('..')
from datasets import utils
from datasets.ravdess import Ravdess
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from multiprocessing import Manager


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
    batch_size=64 , shuffle=True, num_workers=5,pin_memory=True
)

for x,y in dataloader:
    print (x )
    break