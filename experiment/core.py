from sklearn.model_selection import KFold
from torch.utils import data
from torch.utils.data import Subset
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import Dataset
import torchmetrics


import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


import json

class Experiment:

    def __init__(self, models , dataset, dataset_args_train , dataset_args_test ,k=4,n=10 , name="experiment"):

        timestamp = str(time.time()).split(".")[0]

        self.name = name + "_" + timestamp

        # Models  [  (m_class, config) , ( )  ... ]
        self.models = models
        
        # print(dataset_args_train)
        # Dataset 
        self.dataset_train = dataset( *dataset_args_train )
        self.dataset_test = dataset( *dataset_args_test )

        # Args
        self.k = k 
        self.n = n 

        self.data = {
            "name": self.name,
            "real_name": name,
            "k": self.k,
            "n": self.n,
            "iterations": []
        }

    def store(self):
        with open( "clean_logs/"+self.name+".json" , "w" ) as write:
            json.dump( self.data , write )

    def recolect(self):

        for i in range(self.n):
            folds = []
            kf = KFold(n_splits=self.k)
            t = time.time()

            for train_index, test_index in kf.split(self.dataset_train): 
                k_fold_res = []

                train = Subset(self.dataset_train,train_index)
                test = Subset(self.dataset_test,test_index)
                
                for model_class, config in self.models:
                
                    train_dataloader = DataLoader( train ,
                        batch_size=config["batch_size"] , shuffle=True, num_workers=3 , persistent_workers=True
                    )

                    test_dataloader = DataLoader( test ,
                        batch_size=config["batch_size"] , shuffle=False, num_workers=3 , persistent_workers=True,
                    )

                    net = model_class(config["lr"])
                    
                    trainer = Trainer(gpus=1,max_epochs=config["epochs"],precision=16,enable_progress_bar=False)
                    trainer.fit(net,train_dataloader)

                    result = trainer.test(net, dataloaders=test_dataloader, verbose=False )[0]

                    k_fold_res.append({
                        "architecture":  config["architecture"],
                        "metrics" : result
                    })

                    print ( config["architecture"] , result["test/acc"] , end="\t")
                
                folds.append( k_fold_res )
                print("")
            
            self.data["iterations"].append( {
                "time":  time.time() - t ,
                "folds": folds
            })
            print ( "End Iteration {} in {} seconds \n".format(i, time.time() - t))
            self.store()
            





            

