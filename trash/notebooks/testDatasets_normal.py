import sys
sys.path.append('..')

import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from datasets.ravdess import Ravdess
from datasets.utils import collate_fn, melspectrogram, randomcrop, plot_spectrogram, powertodb

from multiprocessing import Manager


dataset = Ravdess(
    "../ser_datasets/ravdess/train.csv",
    "../ser_datasets/ravdess/audios",
    transform = [randomcrop],
    
)


dataloader = DataLoader( dataset ,
    batch_size=128 , shuffle=True, num_workers = 3 , persistent_workers=True, 
)

for i in range(100):
    for x,y in dataloader:
        print(x.shape)
    print("i end",i)

