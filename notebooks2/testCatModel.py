from models.dim.cnn import CNNDim
from datasets.kusisqadim import KusisqaDim
from datasets.utils import randomcrop, centercrop
from torch.utils.data import DataLoader
import torch

net = CNNDim.load_from_checkpoint("logs/ser_kusisqaDim/3oan6uts/checkpoints/epoch=199-step=9599.ckpt",lr=0.001,out=3)


dataset = KusisqaDim(
        "ser_datasets/kusisqadim/train.csv",
        "ser_datasets/kusisqadim/audios",
        transform=[centercrop]
    )

dataloader = DataLoader( dataset ,
    batch_size=16 , shuffle=True, num_workers = 3 , persistent_workers=True, 
)

net.eval()
with torch.no_grad():

    for x,y in dataloader:
        y_p = net(x)
        print(y_p.shape)

        for i in range( len(y)):
            print ( y[i] , y_p[i] )
        break