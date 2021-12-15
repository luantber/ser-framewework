from tasks import kusisqadim_task
from models.cnn_dim import CNNDim

config = dict(lr=0.002, epochs=32, batch_size=32, loss="mse" , architecture="CNNDim")
kusisqadim_task.run(CNNDim, config)
