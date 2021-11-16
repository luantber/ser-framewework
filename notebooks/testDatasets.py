import sys
sys.path.append('..')

import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from datasets.ravdess import Ravdess
from datasets.utils import collate_fn, melspectrogram, randomcrop, plot_spectrogram, powertodb

from multiprocessing import Manager
manager = Manager()
shared_dict1 = manager.dict()
shared_dict2 = manager.dict()

dataset = Ravdess(
    "../ser_datasets/ravdess/train.csv",
    "../ser_datasets/ravdess/audios",
    shared_dict1,
    shared_dict2,
    transform = [randomcrop]
)


dataloader = DataLoader( dataset ,
    batch_size=128 , shuffle=True,
)

for i in range(150):
    for x,y in dataloader:
        print(x.shape)
    print("i end",i)

