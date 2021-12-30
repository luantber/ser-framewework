from tasks import kusisqaDimTask
from models.cnn_dim import CNNDim
from models.resnet import Resnet

if __name__ == "__main__":

    # # MSE
    # config = dict(
    #     lr=0.002, epochs=32, batch_size=32, loss="mse", architecture="CNNDim"
    # )
    # kusisqaDimTask.run(CNNDim, config)

    # # CCC
    # config = dict(
    #     lr=0.002, epochs=32, batch_size=32, loss="ccc", architecture="CNNDim"
    # )
    # kusisqaDimTask.run(CNNDim, config)
    # MSE
    # config = dict(
    #     lr=0.001, epochs=128, batch_size=64, loss="mse", architecture="CNNDim"
    # )
    # kusisqaDimTask.run(CNNDim, config)

    # CCC
    # config = dict(
    #     lr=0.001, epochs=300, batch_size=128, loss="ccc", architecture="Resnet"
    # )
    # kusisqaDimTask.run(Resnet, config)

    config = dict(
      lr=0.0008, epochs=300, batch_size=128, loss="ccc", architecture="CNNDim"
    )
    kusisqaDimTask.run(CNNDim, config)
