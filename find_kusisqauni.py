from tasks import kusisqadim_uni
from models.cnn_uni import CNNUniV1

from pytorch_lightning import seed_everything

seed_everything(7)


# config = dict(lr=0.002, out=2, epochs=32, batch_size=32, architecture="CNN")
# kusisqadim_uni.run(CNNUniV1, config, dimension=1)


# config = dict(lr=0.002, out=2, epochs=32, batch_size=32, architecture="CNN")
# kusisqadim_uni.run(CNNUniV1, config, dimension=2)


# config = dict(lr=0.002, out=2, epochs=32, batch_size=32, architecture="CNN")
# kusisqadim_uni.run(CNNUniV1, config, dimension=3)

# kusisqadim_uni.test_reproducibility(dimension=1)
