from tasks import kusisqaDimTask
from models.cnn_dim import CNNDim

if __name__ == "__main__":
    config = dict(
        lr=0.002, epochs=32, batch_size=32, out=3, loss="mse", architecture="CNNDim"
    )
    kusisqaDimTask.run(CNNDim, config)
