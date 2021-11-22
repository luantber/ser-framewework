from tasks import kusisqadim_task

from models.dim.cnn import CNNDim



config = dict( 
            lr=0.001,
            out=3,
            epochs=300,
            batch_size=64,
            architecture="CNN1Dim"
)

kusisqadim_task.run( CNNDim , config )
