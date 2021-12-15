from torchmetrics import Metric
from torchmetrics.functional import mean_squared_error
import torch


class MSE(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        v_pred, v_target = preds[:, 0], target[:, 0]
        a_pred, a_target = preds[:, 1], target[:, 2]
        d_pred, d_target = preds[:, 2], target[:, 2]

        mse_v = mean_squared_error(v_pred, v_target)
        mse_a = mean_squared_error(a_pred, a_target)
        mse_d = mean_squared_error(d_pred, d_target)

        a = 0.1
        b = 0.5


        self.correct += a*mse_v + b*mse_a + (1-a-b)*mse_d

        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total
