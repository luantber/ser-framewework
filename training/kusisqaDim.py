from tasks import kusisqaDimTask
from models.cnn_dim import CNNDim

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
    config = dict(
        lr=0.001, epochs=64, batch_size=64, loss="ccc", architecture="CNNDim"
    )
    kusisqaDimTask.run(CNNDim, config)
