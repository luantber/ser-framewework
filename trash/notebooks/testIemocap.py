from datasets.iemocap import Iemocap 
from datasets.utils import randomcrop, centercrop
from torch.utils.data import DataLoader

dataset_train = Iemocap(
        "ser_datasets/iemocap/train.csv",
        "ser_datasets/iemocap/audios",
        transform=[centercrop]
)


train_dataloader = DataLoader( dataset_train ,
        batch_size=64 , shuffle=True, num_workers = 3 , persistent_workers=True, 

)

import matplotlib.pyplot as plt

for x,y in train_dataloader:
    print( x[0].shape , y)
    
    plt.imshow(x[0])

    
    plt.show()