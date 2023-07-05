from typing import Optional, Dict
from pytorch_widedeep.callbacks import Callback, EarlyStopping as ES
import src
import numpy as np
import copy


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
        if epoch % src.setting["verbose_per_epoch"] == 0 and self.verbose:
            print(
                f"Epoch: {epoch + 1}/{self.total_epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                f"Min val loss: {np.min(self.val_ls):.4f}"
            )


class EarlyStopping(ES):
    # See this issue https://github.com/jrzaurin/pytorch-widedeep/issues/175
    def on_epoch_end(
        self, epoch: int, logs: Optional[Dict] = None, metric: Optional[float] = None
    ):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.state_dict = copy.deepcopy(self.model.state_dict())
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.trainer.early_stop = True

    def on_train_end(self, logs: Optional[Dict] = None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
        if self.restore_best_weights and self.state_dict is not None:
            if self.verbose > 0:
                print("Restoring model weights from the end of the best epoch")
            self.model.load_state_dict(self.state_dict)
