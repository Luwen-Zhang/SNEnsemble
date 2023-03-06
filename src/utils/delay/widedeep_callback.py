import numpy as np
from typing import Optional, Dict
from pytorch_widedeep.callbacks import Callback


class WideDeepCallback(Callback):
    def __init__(self, total_epoch, verbose):
        super(WideDeepCallback, self).__init__()
        self.val_ls = []
        self.total_epoch = total_epoch
        self.verbose = verbose

    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[Dict] = None,
        metric: Optional[float] = None,
    ):
        train_loss = logs["train_loss"]
        val_loss = logs["val_loss"]
        self.val_ls.append(val_loss)
        if epoch % 20 == 0 and self.verbose:
            print(
                f"Epoch: {epoch + 1}/{self.total_epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                f"Min val loss: {np.min(self.val_ls):.4f}"
            )
