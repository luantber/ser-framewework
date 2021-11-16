import sys
sys.path.append('..')

import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from datasets.ravdess import Ravdess
from datasets.utils import collate_fn, melspectrogram, randomcrop, plot_spectrogram, powertodb


data = Ravdess(
    "../ser_datasets/ravdess/train.csv",
    "../ser_datasets/ravdess/audios",
    transform = [melspectrogram,powertodb,randomcrop]
    )

sr = data.get_sr()


dataloader = DataLoader( data ,
    batch_size=32 , shuffle=True,
)

ys = []
for x,y in dataloader:
    # print( x[0].shape )   
    # plt.imshow(x[0])
    # break
    ys += y.tolist()

# print(ys)

from collections import Counter
a = dict(Counter(ys))
print(a)

plt.hist(ys)
plt.show()