from tasks import kusisqadim_task
from models.dim.cnn import CNNDim

config = dict( 
            lr=0.002,
            out=3,
            epochs=32,
            batch_size=32,
            architecture="CNN1Dim"
)

kusisqadim_task.run( CNNDim , config )
