from datasets.kusisqauni import KusisqaUni
from datasets.utils import randomcrop, centercrop

from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader


from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb
import numpy as np


def get_datasets(dimension1, dimension2=None, prefix=""):
    dataset_train = KusisqaUni(
        prefix + "ser_datasets/kusisqadim/train.csv",
        prefix + "ser_datasets/kusisqadim/audios",
        dimension1=dimension1,
        dimension2=dimension2,
        transform=[randomcrop],
    )

    dataset_test = KusisqaUni(
        prefix + "ser_datasets/kusisqadim/train.csv",
        prefix + "ser_datasets/kusisqadim/audios",
        dimension1=dimension1,
        dimension2=dimension2,
        transform=[centercrop],
    )

    print(len(dataset_train))
    indexes = np.arange(len(dataset_train))

    ## Shuffle the indexes
    np.random.seed(42)
    np.random.shuffle(indexes)

    print(">get", indexes)
    split = int(len(indexes) * 0.8)

    train, test = indexes[:split], indexes[split:]

    assert len(train) + len(test) == len(indexes)

    train_subset = Subset(dataset_train, train)
    test_subset = Subset(dataset_test, test)

    return train_subset, test_subset


def run(model, config, dimension1, dimension2, prefix=""):

    train_subset, test_subset = get_datasets(dimension1, dimension2, prefix=prefix)

    train_dataloader = DataLoader(
        train_subset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=3,
        persistent_workers=True,
    )

    test_dataloader = DataLoader(
        test_subset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=3,
        persistent_workers=True,
    )

    config["dimension"] = str(dimension1) + "_" + str(dimension2)

    ## Training
    wandb_logger = WandbLogger(
        project="ser_kusisqaUniV2",
        config=config,
        save_dir=prefix + "logs/",
        log_model=True,
    )

    net = model(config["lr"], config["out"])
    trainer = Trainer(
        gpus=1,
        logger=wandb_logger,
        max_epochs=config["epochs"],
        precision=16,
    )
    trainer.fit(net, train_dataloader, test_dataloader)

    wandb.finish()
