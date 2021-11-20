from tasks import ravdess_task
from models.cnn import CNN
from models.cnn2 import CNN2
from models.cnn3 import CNN3



## CNN1

config = dict( 
            lr=0.0005,
            out=8,
            epochs=300,
            batch_size=64,
            architecture="CNN1"
        )
ravdess_task.run( CNN , config )

## CNN2

config = dict( 
            lr=0.0005,
            out=8,
            epochs=250,
            batch_size=64,
            architecture="CNN2"
        )
ravdess_task.run( CNN2 , config )


## CNN3

config = dict( 
            lr=0.0005,
            out=8,
            epochs=250,
            batch_size=64,
            architecture="CNN3"
        )
ravdess_task.run( CNN3 , config )
