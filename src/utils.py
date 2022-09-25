import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import scipy.stats as st
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

sns.reset_defaults()

matplotlib.rc("text", usetex=True)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["figure.autolayout"] = True
# plt.rcParams["legend.frameon"] = False

def split_by_material(dataset, mat_lay, mat_lay_set, train_val_test, validation):
    def mat_lay_index(chosen_mat_lay, mat_lay):
        index = []
        for material in chosen_mat_lay:
            where_material = np.where(mat_lay == material)[0]
            index += list(where_material)
        return np.array(index)

    if validation:
        train_mat_lay, test_mat_lay = train_test_split(mat_lay_set, test_size=train_val_test[2], random_state=0)
        train_mat_lay, val_mat_lay = train_test_split(train_mat_lay,
                                                      test_size=train_val_test[1] / np.sum(train_val_test[0:2]), random_state=0)
        train_dataset = Subset(dataset,mat_lay_index(train_mat_lay, mat_lay))
        val_dataset = Subset(dataset,mat_lay_index(val_mat_lay, mat_lay))
        test_dataset = Subset(dataset,mat_lay_index(test_mat_lay, mat_lay))


        df = pd.concat([pd.DataFrame({'train material': train_mat_lay}),
                        pd.DataFrame({'val material': val_mat_lay}),
                        pd.DataFrame({'test material': test_mat_lay})], axis=1)

        df.to_excel('../output/material_split.xlsx', engine='openpyxl', index=False)
        return train_dataset, val_dataset, test_dataset
    else:
        train_mat_lay, test_mat_lay = train_test_split(mat_lay_set, test_size=train_val_test[1], random_state=0)

        train_dataset = Subset(dataset, mat_lay_index(train_mat_lay, mat_lay))
        test_dataset = Subset(dataset, mat_lay_index(test_mat_lay, mat_lay))

        df = pd.concat([pd.DataFrame({'train material': train_mat_lay}),
                        pd.DataFrame({'test material': test_mat_lay})], axis=1)
        df.to_excel('../output/material_split.xlsx', engine='openpyxl', index=False)
        return train_dataset, test_dataset



def replace_column_name(df, name_mapping):
    columns = list(df.columns)
    for idx in range(len(columns)):
        try:
            columns[idx] = name_mapping[columns[idx]]
        except:
            pass
    df_tmp = df.copy()
    df_tmp.columns = columns
    return df_tmp


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def plot_truth_pred(ax, ground_truth, prediction, **kargs):
    ax.scatter(ground_truth, prediction, **kargs)
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")


def plot_absence_ratio(ax, df_presence, **kargs):
    ax.set_axisbelow(True)
    x = df_presence["feature"].values
    y = df_presence["ratio"].values

    # ax.set_facecolor((0.97,0.97,0.97))
    # plt.grid(axis='x')
    plt.grid(axis="x", linewidth=0.2)
    # plt.barh(x,y, color= [clr_map[name] for name in x])
    sns.barplot(y, x, **kargs)
    ax.set_xlim([0, 1])
    ax.set_xlabel("Data absence ratio")


def plot_importance(ax, features, attr, **kargs):
    df = pd.DataFrame(columns=["feature", "attr"])
    df["feature"] = features
    df["attr"] = np.abs(attr) / np.sum(np.abs(attr))
    df.sort_values(by="attr", inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)

    ax.set_axisbelow(True)
    x = df["feature"].values
    y = df["attr"].values

    # ax.set_facecolor((0.97,0.97,0.97))
    # plt.grid(axis='x')
    plt.grid(axis="x", linewidth=0.2)
    # plt.barh(x,y, color= [clr_map[name] for name in x])
    sns.barplot(y, x, **kargs)
    # ax.set_xlim([0, 1])
    ax.set_xlabel("Permutation feature importance")


def calculate_absence_ratio(df_tmp):
    df_presence = pd.DataFrame(columns=["feature", "ratio"])

    for column in df_tmp.columns:
        presence = len(np.where(df_tmp[column].notna())[0])
        # print(f'{column},\t\t {presence}/{len(df_all[column])}, {presence/len(df_all[column]):.3f}')

        df_presence = pd.concat(
            [
                df_presence,
                pd.DataFrame(
                    {"feature": column, "ratio": 1 - presence / len(df_tmp[column])},
                    index=[0],
                ),
            ],
            axis=0,
            ignore_index=True,
        )

    df_presence.sort_values(by="ratio", inplace=True, ascending=False)
    df_presence.reset_index(drop=True, inplace=True)

    # df_presence.drop([0, 1, 2, 5, 9, 10, 11, 13, 14, 15, 19, 23, ])

    return df_presence


def calculate_pdp(model, feature_data, feature_idx, grid_size=100):
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
        model_predictions.append(np.mean(model(X_pdp).cpu().detach().numpy()))

    model_predictions = np.array(model_predictions)

    return x_values, model_predictions


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
