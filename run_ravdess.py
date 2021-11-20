from experiment.core import Experiment
from models.cnn import CNN
from models.cnn2 import CNN2
from models.cnn3 import CNN3

from datasets.ravdess import Ravdess
from datasets.utils import randomcrop, centercrop

## CNN1 300
configA = dict( lr=0.0005,out=8,epochs=300, batch_size=64, architecture="CNN1" )
modelA = ( CNN , configA )

## CNN2 250
configB = dict( lr=0.0005,out=8, epochs=250, batch_size=64, architecture="CNN2" )
modelB = (CNN2 , configB)

## CNN3 250
configC = dict( lr=0.0005,out=8, epochs=250, batch_size=64, architecture="CNN3" )
modelC = (CNN3 , configC)

ravdess_exp = Experiment( 
    [ modelA , modelB, modelC ],

    Ravdess,

    (
        "ser_datasets/ravdess/train.csv",
        "ser_datasets/ravdess/audios",
        [randomcrop]
    ),

    ( 
        "ser_datasets/ravdess/train.csv",
        "ser_datasets/ravdess/audios",
        [centercrop]
    ),
    k=4,
    n=2,
    name= "ravdess_f1acc_k4"
)

ravdess_exp.recolect()