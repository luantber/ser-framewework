"""
  Only for overfitting reasons
"""

from tasks import kusisqaDimTaskOneBatch
from models.cnn_dim import CNNDim
from models.resnet import Resnet


if __name__ == "__main__":

    # # CCC
    config = dict(
        lr=0.001, epochs=300, batch_size=80, loss="ccc", architecture="CNNDim"
    )
    kusisqaDimTaskOneBatch.run(Resnet, config)
