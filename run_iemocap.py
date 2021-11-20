from experiment.core import Experiment
from models.cnn import CNN
from models.cnn2 import CNN2
from models.cnn3 import CNN3

from datasets.iemocap import Iemocap
from datasets.utils import randomcrop, centercrop

## CNN1 300
configA = dict( lr=0.001,out=5,epochs=140, batch_size=256, architecture="CNN1" )
modelA = ( CNN , configA )

## CNN2 250
configB = dict( lr=0.001,out=5, epochs=160, batch_size=256, architecture="CNN2" )
modelB = (CNN2 , configB)

## CNN3 250
configC = dict( lr=0.001,out=5, epochs=180, batch_size=256, architecture="CNN3" )
modelC = (CNN3 , configC)

Iemocap_exp = Experiment( 
    [ modelA , modelB, modelC ],

    Iemocap,

    (
        "ser_datasets/iemocap/train.csv",
        "ser_datasets/iemocap/audios",
        [randomcrop]
    ),

    ( 
        "ser_datasets/iemocap/train.csv",
        "ser_datasets/iemocap/audios",
        [centercrop]
    ),
    k=5,
    n=30,
    name= "iemocap_f1acc_k5"
)

Iemocap_exp.recolect()