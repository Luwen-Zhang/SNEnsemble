'''
All utilities used in the project.
'''

###############################################
# Xueling Luo @ Shanghai Jiao Tong University #
###############################################
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from torch.utils.data import Subset
import torch.utils.data as Data
from torch.nn import functional as F
from importlib import import_module
from skopt.space import Real, Integer, Categorical

clr = sns.color_palette("deep")


def load_config(path):
    data = import_module(path).config.data

    tmp_static_params = {}
    for key in data['static_params']:
        tmp_static_params[key] = data[key]
    data['static_params'] = tmp_static_params

    tmp_chosen_params = {}
    for key in data['chosen_params']:
        tmp_chosen_params[key] = data[key]
    data['chosen_params'] = tmp_chosen_params

    key_chosen = list(data['chosen_params'].keys())
    key_space = list(data['SPACEs'].keys())
    for a, b in zip(key_chosen, key_space):
        if a != b:
            raise Exception('Variables in \'chosen_params\' and \'SPACEs\' should be in the same order.')

    SPACE = []
    for var in key_space:
        setting = data['SPACEs'][var]
        type = setting['type']
        setting.pop('type')
        if type == 'Real':
            SPACE.append(Real(**setting))
        elif type == 'Categorical':
            SPACE.append(Categorical(**setting))
        elif type == 'Integer':
            SPACE.append(Integer(**setting))
        else:
            raise Exception('Invalid type of skopt space.')
    data['SPACE'] = SPACE

    return data


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


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

sns.reset_defaults()

matplotlib.rc("text", usetex=True)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["figure.autolayout"] = True


# plt.rcParams["legend.frameon"] = False

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)


class NN(nn.Module):
    def __init__(self, n_inputs, n_outputs, layers):
        super(NN, self).__init__()
        num_inputs = n_inputs
        num_hiddens1 = layers[0]
        num_hiddens2 = layers[1]
        num_hiddens3 = layers[2]
        num_outputs = n_outputs
        self.net = nn.Sequential()
        self.net.add_module("input", nn.Linear(num_inputs, layers[0]))
        self.net.add_module("ReLU", nn.ReLU())
        for idx in range(len(layers) - 1):
            self.net.add_module(str(idx), nn.Linear(layers[idx], layers[idx + 1]))
            self.net.add_module("ReLU" + str(idx), nn.ReLU())
        self.net.add_module("output", nn.Linear(layers[-1], num_outputs))

        self.net.apply(init_weights)

    def forward(self, x):
        x = self.net(x)
        return x


def train(model, train_loader, optimizer, loss_fn):
    model.train()
    avg_loss = 0
    for idx, (data, yhat) in enumerate(train_loader):
        optimizer.zero_grad()
        y = model(data)
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
        for idx, (data, yhat) in enumerate(test_loader):
            y = model(data)
            loss = loss_fn(yhat, y)
            avg_loss += loss.item() * len(y)
            pred += list(y.cpu().detach().numpy())
            truth += list(yhat.cpu().detach().numpy())
        avg_loss /= len(test_loader.dataset)
    return np.array(pred), np.array(truth), avg_loss


def test_tensor(test_tensor, test_label_tensor, model, loss_fn):
    model.eval()
    pred = []
    truth = []
    with torch.no_grad():
        y = model(test_tensor)
        loss = loss_fn(test_label_tensor, y)
    return (
        y.cpu().detach().numpy(),
        test_label_tensor.cpu().detach().numpy(),
        loss.item(),
    )


def model_train(
        train_dataset,
        test_dataset,
        val_dataset,
        layers,
        validation,
        loss_fn,
        ckp_path,
        device,
        model=None,
        verbose=True,
        return_loss_list=True,
        verbose_per_epoch=100,
        **params,
):
    # print(params)
    if model is None:
        model = NN(
            train_dataset.dataset.tensors[0].shape[1],
            train_dataset.dataset.tensors[1].shape[1],
            layers,
        ).to(device)

    train_loader = Data.DataLoader(
        train_dataset,
        batch_size=int(params["batch_size"]),
        generator=torch.Generator().manual_seed(0)
    )
    if validation:
        val_loader = Data.DataLoader(
            val_dataset,
            batch_size=len(val_dataset),
            generator=torch.Generator().manual_seed(0)
        )
    else:
        val_loader = None
    test_loader = Data.DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        generator=torch.Generator().manual_seed(0),
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )

    train_ls = []
    val_ls = []
    stop_epoch = params["epoch"]

    early_stopping = EarlyStopping(
        patience=params["patience"], verbose=False, path=ckp_path
    )

    for epoch in range(params["epoch"]):
        train_loss = train(model, train_loader, optimizer, loss_fn)
        train_ls.append(train_loss)
        if validation:
            _, _, val_loss = test(model, val_loader, loss_fn)
            val_ls.append(val_loss)

        if verbose and ((epoch + 1) % verbose_per_epoch == 0 or epoch == 0):
            if validation:
                print(
                    f"Epoch: {epoch + 1}/{stop_epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}"
                )
            else:
                print(f"Epoch: {epoch + 1}/{stop_epoch}, Train loss: {train_loss:.4f}")

        if validation:
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                if verbose:
                    idx = val_ls.index(min(val_ls))
                    print(
                        f"Early stopping at epoch {epoch + 1}, Checkpoint at epoch {idx + 1}, Train loss: {train_ls[idx]:.4f}, Val loss: {val_ls[idx]:.4f}"
                    )
                stop_epoch = epoch + 1
                break
        else:
            early_stopping(train_loss, model)

            if early_stopping.early_stop:
                if verbose:
                    idx = train_ls.index(min(train_ls))
                    print(
                        f"Early stopping at epoch {epoch + 1}, Checkpoint at epoch {idx + 1}, Train loss: {train_ls[idx]:.4f}"
                    )
                stop_epoch = epoch + 1
                break
    if validation:
        idx = val_ls.index(min(val_ls))
        min_loss = val_ls[idx]
    else:
        min_loss = train_ls[-1]

    if return_loss_list:
        return min_loss, train_ls, val_ls
    else:
        return min_loss


class PI_MSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = (
                self.mse(yhat, y)
                + self.eps
                + torch.sum(F.relu(-y)) / len(y)
                + torch.sum(F.relu(y - 5)) / len(y)
        )
        return loss


def split_dataset(data, feature_names, label_name, device, validation, split_by):
    tmp_data = (
        data[feature_names + label_name + ["Material_Code"]].copy().dropna(axis=0)
    )

    mat_lay = tmp_data['Material_Code'].copy()
    mat_lay_set = list(sorted(set(mat_lay)))

    data = data[feature_names + label_name].dropna(axis=0)
    feature_data = data[feature_names]
    label_data = np.log10(data[label_name])

    X = torch.tensor(feature_data.values, dtype=torch.float32).to(device)
    y = torch.tensor(label_data.values, dtype=torch.float32).to(device)
    dataset = Data.TensorDataset(X, y)

    if validation:
        train_val_test = np.array([0.6, 0.2, 0.2])
        if split_by == "random":
            train_size = np.floor(len(label_data) * train_val_test[0]).astype(int)
            val_size = np.floor(len(label_data) * train_val_test[1]).astype(int)
            test_size = len(label_data) - train_size - val_size
            train_dataset, val_dataset, test_dataset = Data.random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(0),
            )
        elif split_by == "material":
            train_dataset, val_dataset, test_dataset = split_by_material(
                dataset, mat_lay, mat_lay_set, train_val_test, validation
            )
        else:
            raise Exception("Split type not implemented")

        print("Dataset size:", len(train_dataset), len(val_dataset), len(test_dataset))
    else:
        train_test = np.array([0.8, 0.2])
        if split_by == "random":
            train_size = np.floor(len(label_data) * train_test[0]).astype(int)
            test_size = len(label_data) - train_size
            train_dataset, test_dataset = Data.random_split(
                dataset,
                [train_size, test_size],
                generator=torch.Generator().manual_seed(0),
            )
        elif split_by == "material":
            train_dataset, test_dataset = split_by_material(
                dataset, mat_lay, mat_lay_set, train_test, validation
            )
        else:
            raise Exception("Split type not implemented")
        val_dataset = None
        print("Dataset size:", len(train_dataset), len(test_dataset))

    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    scaler.fit(train_dataset.dataset.tensors[0].cpu().numpy()[train_dataset.indices, :])
    # torch.data.Dataset.Subset share the same memory, so only transform once.
    transformed = scaler.transform(train_dataset.dataset.tensors[0].cpu().numpy())
    train_dataset.dataset.tensors = (
        torch.tensor(transformed, dtype=torch.float32).to(device),
        train_dataset.dataset.tensors[1],
    )
    X = torch.tensor(scaler.transform(X.cpu().numpy()), dtype=torch.float32).to(device)

    return (
        feature_data,
        label_data,
        X,
        y,
        train_dataset,
        val_dataset,
        test_dataset,
        scaler,
    )


def split_by_material(dataset, mat_lay, mat_lay_set, train_val_test, validation):
    def mat_lay_index(chosen_mat_lay, mat_lay):
        index = []
        for material in chosen_mat_lay:
            where_material = np.where(mat_lay == material)[0]
            index += list(where_material)
        return np.array(index)

    if validation:
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

        df.to_excel("../output/material_split.xlsx", engine="openpyxl", index=False)
        return train_dataset, val_dataset, test_dataset
    else:
        train_mat_lay, test_mat_lay = train_test_split(
            mat_lay_set, test_size=train_val_test[1], shuffle=False
        )

        train_dataset = Subset(dataset, mat_lay_index(train_mat_lay, mat_lay))
        test_dataset = Subset(dataset, mat_lay_index(test_mat_lay, mat_lay))

        df = pd.concat(
            [
                pd.DataFrame({"train material": train_mat_lay}),
                pd.DataFrame({"test material": test_mat_lay}),
            ],
            axis=1,
        )
        df.to_excel("../output/material_split.xlsx", engine="openpyxl", index=False)
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

    palette = df['pal']

    # ax.set_facecolor((0.97,0.97,0.97))
    # plt.grid(axis='x')
    plt.grid(axis="x", linewidth=0.2)
    # plt.barh(x,y, color= [clr_map[name] for name in x])
    sns.barplot(y, x, palette=palette, **kargs)
    # ax.set_xlim([0, 1])
    ax.set_xlabel("Permutation feature importance")

    from matplotlib.patches import Patch, Rectangle

    legend = ax.legend(handles=[Rectangle((0, 0), 1, 1, color=value, ec='k', label=key) for key, value in
                                zip(clr_map.keys(), clr_map.values())],
                       loc='lower right', handleheight=2, fancybox=False, frameon=False)

    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor([1, 1, 1, .4])


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


def plot_loss(train_ls, val_ls, ax):
    ax.plot(
        np.arange(len(train_ls)),
        train_ls,
        label="Train loss",
        linewidth=2,
        color=clr[0],
    )
    if len(val_ls) > 0:
        ax.plot(
            np.arange(len(val_ls)),
            val_ls,
            label="Validation loss",
            linewidth=2,
            color=clr[1],
        )
    # minposs = val_ls.index(min(val_ls))+1
    # ax.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    ax.legend()
    ax.set_ylabel("MSE Loss")
    ax.set_xlabel("Epoch")


def plot_truth_pred_NN(train_dataset, val_dataset, test_dataset, model, loss_fn, ax):
    train_loader = Data.DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        generator=torch.Generator().manual_seed(0),
    )
    if val_dataset is not None:
        val_loader = Data.DataLoader(
            val_dataset,
            batch_size=len(val_dataset),
            generator=torch.Generator().manual_seed(0),
        )
    test_loader = Data.DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        generator=torch.Generator().manual_seed(0),
    )

    prediction, ground_truth, loss = test(model, train_loader, loss_fn)
    r2 = r2_score(ground_truth, prediction)
    print(f"Train MSE Loss: {loss:.4f}, R2: {r2:.4f}")
    plot_truth_pred(
        ax,
        10 ** ground_truth,
        10 ** prediction,
        s=20,
        color=clr[0],
        label=f"Train dataset ($R^2$={r2:.3f})",
        linewidth=0.4,
        edgecolors="k",
    )

    if val_dataset is not None:
        prediction, ground_truth, loss = test(model, val_loader, loss_fn)
        r2 = r2_score(ground_truth, prediction)
        print(f"Validation MSE Loss: {loss:.4f}, R2: {r2:.4f}")
        plot_truth_pred(
            ax,
            10 ** ground_truth,
            10 ** prediction,
            s=20,
            color=clr[2],
            label=f"Val dataset ($R^2$={r2:.3f})",
            linewidth=0.4,
            edgecolors="k",
        )

    prediction, ground_truth, loss = test(model, test_loader, loss_fn)
    r2 = r2_score(ground_truth, prediction)
    print(f"Test MSE Loss: {loss:.4f}, R2: {r2:.4f}")
    plot_truth_pred(
        ax,
        10 ** ground_truth,
        10 ** prediction,
        s=20,
        color=clr[1],
        label=f"Test dataset ($R^2$={r2:.3f})",
        linewidth=0.4,
        edgecolors="k",
    )

    set_truth_pred(ax)


def plot_truth_pred_sklearn(
        feature_data, label_data, train_indices, test_indices, ax, model, split_by="random"
):
    train_x = feature_data.values[np.array(train_indices), :]
    train_y = label_data.values[np.array(train_indices), :]

    test_x = feature_data.values[np.array(test_indices), :]
    test_y = label_data.values[np.array(test_indices), :]

    if train_y.shape[1] == 1:
        train_y = train_y[:, 0]
        test_y = test_y[:, 0]

    model.fit(train_x, train_y)

    pred_y = model.predict(train_x)
    r2 = r2_score(train_y, pred_y)
    loss = np.mean((train_y - pred_y) ** 2)
    print(f"Train MSE Loss: {loss:.4f}, R2: {r2:.4f}")
    plot_truth_pred(
        ax,
        10 ** train_y,
        10 ** pred_y,
        s=20,
        color=clr[0],
        label=f"Train dataset ($R^2$={r2:.3f})",
        linewidth=0.4,
        edgecolors="k",
    )

    pred_y = model.predict(test_x)
    r2 = r2_score(test_y, pred_y)
    loss = np.mean((test_y - pred_y) ** 2)
    print(f"Test MSE Loss: {loss:.4f}, R2: {r2:.4f}")
    plot_truth_pred(
        ax,
        10 ** test_y,
        10 ** pred_y,
        s=20,
        color=clr[1],
        label=f"Test dataset ($R^2$={r2:.3f})",
        linewidth=0.4,
        edgecolors="k",
    )

    set_truth_pred(ax)


def plot_pdp(feature_names, x_values_list, mean_pdp_list, X, hist_indices):
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
    print(figsize, width, height)

    fig = plt.figure(figsize=figsize)

    for idx, focus_feature in enumerate(feature_names):
        ax = plt.subplot(height, width, idx + 1)
        # ax.plot(x_values_list[idx], mean_pdp_list[idx], color = clr_map[focus_feature], linewidth = 0.5)
        ax.plot(x_values_list[idx], 10 ** mean_pdp_list[idx], color="k", linewidth=0.7)

        ax.set_title(focus_feature, {"fontsize": 12})
        ax.set_xlim([0, 1])
        ax.set_yscale("log")
        ax.set_ylim([10 ** 2, 10 ** 7])
        locmin = matplotlib.ticker.LogLocator(
            base=10.0, subs=[0.1 * x for x in range(10)], numticks=20
        )
        ax.xaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
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


def set_truth_pred(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(
        np.linspace(0, 10 ** 8, 100),
        np.linspace(0, 10 ** 8, 100),
        "--",
        c="grey",
        alpha=0.2,
    )
    ax.set_aspect("equal", "box")
    locmin = matplotlib.ticker.LogLocator(
        base=10.0, subs=[0.1 * x for x in range(10)], numticks=20
    )

    # ax.set(xlim=[10, 10 ** 6], ylim=[10, 10 ** 6])
    ax.xaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
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
