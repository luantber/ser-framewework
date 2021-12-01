import sys
sys.path.append('..')

from experiment.core import Experiment
from datasets.ravdess import Ravdess
from multiprocessing import Manager
from datasets import utils
from models.cnn import CNN

models = [CNN]

dataset = Ravdess(
    "../ser_datasets/ravdess/train.csv",
    "../ser_datasets/ravdess/audios",
    transform = [utils.randomcrop]
)

e = Experiment(models, dataset , n=1)
e.recolect()
