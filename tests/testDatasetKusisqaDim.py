from datasets.kusisqadim import  KusisqaDim
from datasets.utils import randomcrop, centercrop
from torch.utils.data import DataLoader

dataset_train = KusisqaDim(
        "ser_datasets/kusisqadim/train.csv",
        "ser_datasets/kusisqadim/audios",
        transform=[centercrop]
)


train_dataloader = DataLoader( dataset_train ,
    batch_size=64 , shuffle=True, num_workers = 3 , persistent_workers=True,
)


import matplotlib.pyplot as plt
for x,y in train_dataloader:
    print(x.shape, y.shape )
    print( x[0].shape , y[0])
    plt.imshow(x[0])
    plt.show()
    break