from tasks import ravdess_task
from models.cnn import CNN
from models.cnn2 import CNN2
from models.cnn3 import CNN3


config = dict( 
            lr=0.0005,
            epochs=250,
            batch_size=64,
            architecture="CNN3"
        )


# ravdess_task.run( CNN , config )
ravdess_task.run( CNN3 , config )
