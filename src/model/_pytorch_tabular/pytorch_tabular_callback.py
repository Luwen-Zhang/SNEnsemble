import pytorch_lightning as pl
from pytorch_lightning import Callback
import src
import numpy as np


class PytorchTabularCallback(Callback):
    def __init__(self, verbose, total_epoch):
        super(PytorchTabularCallback, self).__init__()
        self.val_ls = []
        self.verbose = verbose
        self.total_epoch = total_epoch

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        logs = trainer.callback_metrics
        train_loss = logs["train_mean_squared_error"].detach().cpu().numpy()
        val_loss = logs["valid_mean_squared_error"].detach().cpu().numpy()
        self.val_ls.append(val_loss)
        epoch = trainer.current_epoch
        if epoch % src.setting["verbose_per_epoch"] == 0 and self.verbose:
            print(
                f"Epoch: {epoch + 1}/{self.total_epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                f"Min val loss: {np.min(self.val_ls):.4f}"
            )
