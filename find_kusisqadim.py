from tasks import kusisqa_task

from models.dim.cnn import CNNDim



config = dict( 
            lr=0.001,
            out=3,
            epochs=200,
            batch_size=64,
            architecture="CNN1Dim"
)

kusisqa_task.run( CNNDim , config )
