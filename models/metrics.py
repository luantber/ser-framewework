from torchmetrics import Metric
from torchmetrics.functional import mean_squared_error
import torch

from torch.nn.functional import mse_loss
from torch import nn


def ccc(x, y):
    # Pearson
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    p = cos(x - x.mean(), y - y.mean())

    d_x = x.std()
    d_y = y.std()

    mu_x = x.mean()
    mu_y = y.mean()

    return (2 * p * d_x * d_y) / (d_x ** 2 + d_y ** 2 + (mu_x - mu_y) ** 2)


def ccc_loss_custom(preds, target):
    v_pred, v_target = preds[:, 0], target[:, 0]
    a_pred, a_target = preds[:, 1], target[:, 1]
    d_pred, d_target = preds[:, 2], target[:, 2]

    lccc_v = 1 - ccc(v_pred, v_target)
    lccc_a = 1 - ccc(a_pred, a_target)
    lccc_d = 1 - ccc(d_pred, d_target)

    a = torch.tensor(0.1)
    b = torch.tensor(0.5)

    return a * lccc_v + b * lccc_a + (1 - a - b) * lccc_d


class CCC(Metric):
    """
    CCC Metric
    """

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        self.correct += 1 - ccc_loss_custom(preds, target)
        self.total += 1

    def compute(self):
        return self.correct.float() / self.total


def mse_loss_custom(preds, target):
    v_pred, v_target = preds[:, 0], target[:, 0]
    a_pred, a_target = preds[:, 1], target[:, 1]
    d_pred, d_target = preds[:, 2], target[:, 2]

    mse_v = mse_loss(v_pred, v_target)
    mse_a = mse_loss(a_pred, a_target)
    mse_d = mse_loss(d_pred, d_target)

    a = torch.tensor(0.1)
    b = torch.tensor(0.5)

    return a * mse_v + b * mse_a + (1 - a - b) * mse_d


class MSE(Metric):
    """
    balanced metrics
    """

    def __init__(self):
        super().__init__()

        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        self.correct += mse_loss_custom(preds, target)
        self.total += 1

    def compute(self):
        return self.correct.float() / self.total
