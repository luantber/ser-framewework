from tasks import iemocap_task
from models.cnn import CNN
from models.cnn2 import CNN2
from models.cnn3 import CNN3


## CNN1

# config = dict( 
#             lr=0.0005,
#             out=6,
#             epochs=300,
#             batch_size=128,
#             architecture="CNN1"
#         )
# iemocap_task.run( CNN , config )

# ## CNN2

# config = dict( 
#             lr=0.0005,
#             out=6,
#             epochs=250,
#             batch_size=64,
#             architecture="CNN2"
#         )
# iemocap_task.run( CNN2 , config )


## CNN3

config = dict( 
            lr=0.001,
            out=5,
            epochs=150,
            batch_size=256,
            architecture="CNN3"
        )
iemocap_task.run( CNN3 , config )
