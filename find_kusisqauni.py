from tasks import kusisqadim_uni
from models.cnn_uni import CNNUniV1


config = dict(lr=0.001, out=2, epochs=64, batch_size=32, architecture="CNN")
kusisqadim_uni.run(CNNUniV1, config, dimension1=1, dimension2=2)

config = dict(lr=0.001, out=2, epochs=64, batch_size=32, architecture="CNN")
kusisqadim_uni.run(CNNUniV1, config, dimension1=2, dimension2=1)
