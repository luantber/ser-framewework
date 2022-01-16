from datasets import KusisqaDim
from datasets.utils import randomcrop, centercrop
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError
from models.cnn_dim import CNNDim
from models.resnet import Resnet
from models.metrics import CCC, ccc

# Plot confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=3)

if __name__ == "__main__":

    dataset_train = KusisqaDim(
        "ser_datasets/test_kusisqa/audio-alegre.csv",
        "ser_datasets/test_kusisqa/Audio-Alegre/",
        transform=[centercrop],
    )

    train_dataloader = DataLoader(
        dataset_train,
        batch_size=32,
        shuffle=False,
        num_workers=3,
        persistent_workers=True,
    )

    path_pesos = "checkpoints/flower.ckpt"

    model = CNNDim.load_from_checkpoint(path_pesos, lr=0.001)
    model.eval()

    metric = MeanAbsoluteError()

    ccc_v = CCC()
    ccc_a = CCC()
    ccc_d = CCC()

    for x, y in train_dataloader:
        resultado = model(x)

        acc = metric(resultado, y)

        mccc_v = ccc(resultado[:, 0], y[:, 0])
        mccc_a = ccc(resultado[:, 1], y[:, 1])
        mccc_d = ccc(resultado[:, 2], y[:, 2])

        ccc_v(mccc_v)
        ccc_a(mccc_a)
        ccc_d(mccc_d)

        for i in range(len(resultado)):
            expected = resultado[i].detach().numpy()
            y_i = y[i].detach().numpy()
            print(expected, "-", y_i, "=", expected - y_i, " ,\t " ,(expected - y_i).mean())

        # print(f"MAE batch: { acc}")

        # print(f"CCC V batch: { mccc_v}")
        # print(f"CCC A batch: { mccc_a}")
        # print(f"CCC D batch: { mccc_d}\n")

    acc = metric.compute()
    mccc_a = ccc_a.compute()
    mccc_v = ccc_v.compute()
    mccc_d = ccc_d.compute()

    print(f"MAE: { acc}")
    print(f"CCC V: { mccc_v}")
    print(f"CCC A : { mccc_a}")
    print(f"CCC D : { mccc_d}")
