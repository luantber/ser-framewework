from tasks import kusisqa_task
from models.cnn import CNN
from models.cnn2 import CNN2
from models.cnn3 import CNN3
from models.cnn3_dropout import CNN3Dropout



number_classes = kusisqa_task.get_number_classes( )

## CNN1
# config = dict( 
#             lr=0.0005,
#             out=8,
#             epochs=300,
#             batch_size=64,
#             architecture="CNN1"
#         )
# kusisqa_task.run( CNN , config )

# ## CNN2
# config = dict( 
#             lr=0.0005,
#             out=8,
#             epochs=250,
#             batch_size=64,
#             architecture="CNN2"
#         )
# kusisqa_task.run( CNN2 , config )


## CNN3
config = dict( 
            lr=0.0008,
            out=number_classes,
            epochs=200,
            batch_size=64,
            architecture="CNN3Dropout"
)

kusisqa_task.run( CNN3Dropout , config )
