"""
All utilities used in the project.
"""

###############################################
# Xueling Luo @ Shanghai Jiao Tong University #
###############################################
import os
import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from src.core.nn_models import *
import logging

clr = sns.color_palette("deep")


def is_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

sns.reset_defaults()

from distutils.spawn import find_executable

if find_executable("latex"):
    matplotlib.rc("text", usetex=True)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["figure.autolayout"] = True


# https://stackoverflow.com/questions/65840698/how-to-make-r2-score-in-nn-lstm-pytorch
def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return 1 - r2


def train(model, train_loader, optimizer, loss_fn):
    model.train()
    avg_loss = 0
    for idx, tensors in enumerate(train_loader):
        optimizer.zero_grad()
        yhat = tensors[-1]
        data = tensors[0]
        additional_tensors = tensors[1 : len(tensors) - 1]
        y = model(data, additional_tensors)
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * len(y)

    avg_loss /= len(train_loader.dataset)
    return avg_loss


def test(model, test_loader, loss_fn):
    model.eval()
    pred = []
    truth = []
    with torch.no_grad():
        # print(test_dataset)
        avg_loss = 0
        for idx, tensors in enumerate(test_loader):
            yhat = tensors[-1]
            data = tensors[0]
            additional_tensors = tensors[1 : len(tensors) - 1]
            y = model(data, additional_tensors)
            loss = loss_fn(yhat, y)
            avg_loss += loss.item() * len(y)
            pred += list(y.cpu().detach().numpy())
            truth += list(yhat.cpu().detach().numpy())
        avg_loss /= len(test_loader.dataset)
    return np.array(pred), np.array(truth), avg_loss


def test_tensor(test_tensor, additional_tensors, test_label_tensor, model, loss_fn):
    model.eval()
    with torch.no_grad():
        y = model(test_tensor, additional_tensors)
        loss = loss_fn(test_label_tensor, y)
    return (
        y.cpu().detach().numpy(),
        test_label_tensor.cpu().detach().numpy(),
        loss.item(),
    )


def split_by_material(dataset, mat_lay, mat_lay_set, train_val_test):
    def mat_lay_index(chosen_mat_lay, mat_lay):
        index = []
        for material in chosen_mat_lay:
            where_material = np.where(mat_lay == material)[0]
            index += list(where_material)
        return np.array(index)

    train_mat_lay, test_mat_lay = train_test_split(
        mat_lay_set, test_size=train_val_test[2], shuffle=False
    )
    train_mat_lay, val_mat_lay = train_test_split(
        train_mat_lay,
        test_size=train_val_test[1] / np.sum(train_val_test[0:2]),
        shuffle=False,
    )
    train_dataset = Subset(dataset, mat_lay_index(train_mat_lay, mat_lay))
    val_dataset = Subset(dataset, mat_lay_index(val_mat_lay, mat_lay))
    test_dataset = Subset(dataset, mat_lay_index(test_mat_lay, mat_lay))

    df = pd.concat(
        [
            pd.DataFrame({"train material": train_mat_lay}),
            pd.DataFrame({"val material": val_mat_lay}),
            pd.DataFrame({"test material": test_mat_lay}),
        ],
        axis=1,
    )

    # df.to_excel("../output/material_split.xlsx", engine="openpyxl", index=False)
    return train_dataset, val_dataset, test_dataset


def plot_importance(ax, features, attr, pal, clr_map, **kargs):
    df = pd.DataFrame(columns=["feature", "attr", "clr"])
    df["feature"] = features
    df["attr"] = np.abs(attr) / np.sum(np.abs(attr))
    df["pal"] = pal
    df.sort_values(by="attr", inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)

    ax.set_axisbelow(True)
    x = df["feature"].values
    y = df["attr"].values

    palette = df["pal"]

    # ax.set_facecolor((0.97,0.97,0.97))
    # plt.grid(axis='x')
    plt.grid(axis="x", linewidth=0.2)
    # plt.barh(x,y, color= [clr_map[name] for name in x])
    sns.barplot(y, x, palette=palette, **kargs)
    # ax.set_xlim([0, 1])
    ax.set_xlabel("Permutation feature importance")

    from matplotlib.patches import Patch, Rectangle

    legend = ax.legend(
        handles=[
            Rectangle((0, 0), 1, 1, color=value, ec="k", label=key)
            for key, value in clr_map.items()
        ],
        loc="lower right",
        handleheight=2,
        fancybox=False,
        frameon=False,
    )

    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor([1, 1, 1, 0.4])


def calculate_pdp(model, feature_data, additional_tensors, feature_idx, grid_size=100):
    x_values = np.linspace(
        np.percentile(feature_data[:, feature_idx].cpu().numpy(), 10),
        np.percentile(feature_data[:, feature_idx].cpu().numpy(), 90),
        grid_size,
    )

    model_predictions = []

    for n in x_values:
        X_pdp = feature_data.clone().detach()
        # X_pdp = resample(X_pdp)
        X_pdp[:, feature_idx] = n
        model_predictions.append(
            np.mean(model(X_pdp, additional_tensors).cpu().detach().numpy())
        )

    model_predictions = np.array(model_predictions)

    return x_values, model_predictions


def plot_pdp(
    feature_names, x_values_list, mean_pdp_list, X, hist_indices, log_trans=True, lower_lim=2, upper_lim=7
):
    max_col = 4
    if len(feature_names) > max_col:
        width = max_col
        if len(feature_names) % max_col == 0:
            height = len(feature_names) // max_col
        else:
            height = len(feature_names) // max_col + 1
        figsize = (14, 3 * height)
    else:
        figsize = (3 * len(feature_names), 2.5)
        width = len(feature_names)
        height = 1
    # print(figsize, width, height)

    fig = plt.figure(figsize=figsize)

    for idx, focus_feature in enumerate(feature_names):
        ax = plt.subplot(height, width, idx + 1)
        # ax.plot(x_values_list[idx], mean_pdp_list[idx], color = clr_map[focus_feature], linewidth = 0.5)
        ax.plot(
            x_values_list[idx],
            10 ** mean_pdp_list[idx] if log_trans else mean_pdp_list[idx],
            color="k",
            linewidth=0.7,
        )

        ax.set_title(focus_feature, {"fontsize": 12})
        ax.set_xlim([0, 1])
        if log_trans:
            ax.set_yscale("log")
            ax.set_ylim([10**lower_lim, 10**upper_lim])
            locmin = matplotlib.ticker.LogLocator(
                base=10.0, subs=[0.1 * x for x in range(10)], numticks=20
            )
            # ax.xaxis.set_minor_locator(locmin)
            ax.yaxis.set_minor_locator(locmin)
            # ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        ax2 = ax.twinx()

        chosen_data = X[hist_indices, idx].cpu().detach().numpy()
        ax2.hist(
            chosen_data,
            bins=x_values_list[idx],
            density=True,
            color=[0, 0, 0],
            alpha=0.2,
            rwidth=0.8,
        )
        # sns.rugplot(data=chosen_data, height=0.05, ax=ax2, color='k')
        # ax2.set_ylim([0,1])
        ax2.set_xlim([np.min(x_values_list[idx]), np.max(x_values_list[idx])])
        ax2.set_yticks([])

    ax = fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.ylabel("Predicted fatigue life")
    plt.xlabel("Value of predictors (standard scaled, $10\%$-$90\%$ percentile)")

    return fig


def plot_partial_err(feature_data, truth, pred, thres=0.8):
    feature_names = list(feature_data.columns)
    max_col = 4
    if len(feature_names) > max_col:
        width = max_col
        if len(feature_names) % max_col == 0:
            height = len(feature_names) // max_col
        else:
            height = len(feature_names) // max_col + 1
        figsize = (14, 3 * height)
    else:
        figsize = (3 * len(feature_names), 2.5)
        width = len(feature_names)
        height = 1
    # print(figsize, width, height)

    fig = plt.figure(figsize=figsize)

    err = np.abs(truth - pred)
    high_err_data = feature_data.loc[np.where(err > thres)[0], :]
    high_err = err[np.where(err > thres)[0]]
    low_err_data = feature_data.loc[np.where(err <= thres)[0], :]
    low_err = err[np.where(err <= thres)[0]]
    for idx, focus_feature in enumerate(feature_names):
        ax = plt.subplot(height, width, idx + 1)
        # ax.plot(x_values_list[idx], mean_pdp_list[idx], color = clr_map[focus_feature], linewidth = 0.5)
        ax.scatter(
            high_err_data[focus_feature].values, high_err, s=1, color=clr[0], marker="s"
        )
        ax.scatter(
            low_err_data[focus_feature].values, low_err, s=1, color=clr[1], marker="^"
        )

        ax.set_title(focus_feature, {"fontsize": 12})

        ax.set_ylim([0, np.max(err) * 1.1])
        ax2 = ax.twinx()

        ax2.hist(
            [high_err_data[focus_feature].values, low_err_data[focus_feature].values],
            bins=np.linspace(
                np.min(feature_data[focus_feature].values),
                np.max(feature_data[focus_feature].values),
                20,
            ),
            density=True,
            color=clr[:2],
            alpha=0.2,
            rwidth=0.8,
        )
        # sns.rugplot(data=chosen_data, height=0.05, ax=ax2, color='k')
        # ax2.set_ylim([0,1])
        # ax2.set_xlim([np.min(x_values_list[idx]), np.max(x_values_list[idx])])
        ax2.set_yticks([])

    ax = fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.ylabel("Prediction absolute error")
    plt.xlabel("Value of predictors")

    return fig


def set_truth_pred(ax, log_trans=True, upper_lim=9):
    if log_trans:
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.plot(
            np.linspace(0, 10**upper_lim, 100),
            np.linspace(0, 10**upper_lim, 100),
            "--",
            c="grey",
            alpha=0.2,
        )
        locmin = matplotlib.ticker.LogLocator(
            base=10.0, subs=[0.1 * x for x in range(10)], numticks=20
        )

        # ax.set_aspect("equal", "box")

        ax.xaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        ax.set_xlim(1, 10**upper_lim)
        ax.set_ylim(1, 10**upper_lim)
        ax.set_box_aspect(1)
    else:
        # ax.set_aspect("equal", "box")
        lx, rx = ax.get_xlim()
        ly, ry = ax.get_ylim()
        l = np.min([lx, ly])
        r = np.max([rx, ry])

        ax.plot(
            np.linspace(l, r, 100),
            np.linspace(l, r, 100),
            "--",
            c="grey",
            alpha=0.2,
        )

        ax.set_xlim(left=l, right=r)
        ax.set_ylim(bottom=l, top=r)
        ax.set_box_aspect(1)

    # ax.set(xlim=[10, 10 ** 6], ylim=[10, 10 ** 6])

    # data_range = [
    #     np.floor(np.min([np.min(ground_truth), np.min(prediction)])),
    #     np.ceil(np.max([np.max(ground_truth), np.max(prediction)]))
    # ]


# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class HiddenPrints:
    def __init__(self, disable_logging=True, disable_std=True):
        self.disable_logging = disable_logging
        self.disable_std = disable_std

    def __enter__(self):
        if self.disable_std:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        if self.disable_logging:
            logging.disable(logging.CRITICAL)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disable_std:
            sys.stdout.close()
            sys.stdout = self._original_stdout
        if self.disable_logging:
            logging.disable(logging.NOTSET)


def disable_tqdm():
    from functools import partialmethod
    from tqdm import tqdm

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    from tqdm.notebook import tqdm

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    from tqdm.autonotebook import tqdm

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    from tqdm.auto import tqdm

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def enable_tqdm():
    from functools import partialmethod
    from tqdm import tqdm

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)
    from tqdm.notebook import tqdm

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)
    from tqdm.autonotebook import tqdm

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)
    from tqdm.auto import tqdm

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, "gettrace") and sys.gettrace() is not None


def gini(x, w=None):
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if len(np.unique(x)) == 1:
        return np.nan
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
