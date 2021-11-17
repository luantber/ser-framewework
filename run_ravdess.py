from experiment.core import Experiment
from models.cnn import CNN
from models.cnn2 import CNN2
from models.cnn3 import CNN3

from datasets.ravdess import Ravdess
from datasets.utils import randomcrop, centercrop

## CNN1
configA = dict( lr=0.0005,epochs=300, batch_size=64, architecture="CNN1" )
modelA = ( CNN , configA )

## CNN2
configB = dict( lr=0.0005, epochs=250, batch_size=64, architecture="CNN2" )
modelB = (CNN2 , configB)

## CNN3
configC = dict( lr=0.0005, epochs=250, batch_size=64, architecture="CNN3" )
modelC = (CNN2 , configB)

ravdess_exp = Experiment( 
    [ modelA , modelB , modelC ],

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
    n=1
)

ravdess_exp.recolect()