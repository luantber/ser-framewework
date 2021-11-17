from sklearn.model_selection import KFold
from torch.utils.data import Subset
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import time 

class Experiment( ):
    def __init__(self,models,dataset,k=4,n=10):
        self.models = models
        self.dataset = dataset
        self.k = k 
        self.n = n 

    def recolect(self):

        for i in range(self.n):
            C_i = []
            kf = KFold(n_splits=self.k)
            t = time.time()

            for train_index, test_index in kf.split(self.dataset): 
                
                train = Subset(self.dataset,train_index)
                test = Subset(self.dataset,test_index)

                net = self.models[0]()

                train_dataloader = DataLoader( train ,
                    batch_size=128 , shuffle=True, num_workers=3 ,pin_memory=True, persistent_workers=True
                )

               
                
                trainer = Trainer(gpus=1,logger=None,max_epochs=180,profiler=None,enable_progress_bar=False)
                trainer.fit(net,train_dataloader)

                # break

                
                # print ( train , test )

            print ( "End Iteration {} in {} seconds \n".format(i, time.time() - t))

