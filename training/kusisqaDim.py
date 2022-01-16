from tasks import kusisqaDimTask
from models.cnn_dim import CNNDim
from models.resnet import Resnet

if __name__ == "__main__":

    config = dict(
        lr=0.001, epochs=250, batch_size=256, loss="ccc", architecture="Resnet"
    )
    kusisqaDimTask.run(Resnet, config)

    config = dict(
        lr=0.001, epochs=250, batch_size=256, loss="ccc", architecture="CNNDim"
    )
    kusisqaDimTask.run(CNNDim, config)

    config = dict(
        lr=0.001, epochs=250, batch_size=256, loss="mse", architecture="Resnet"
    )
    kusisqaDimTask.run(Resnet, config)

    config = dict(
        lr=0.001, epochs=250, batch_size=256, loss="mse", architecture="CNNDim"
    )
    kusisqaDimTask.run(CNNDim, config)
