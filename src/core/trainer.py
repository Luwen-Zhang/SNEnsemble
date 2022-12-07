"""
The basic class for the project. It includes configuration, data processing, plotting,
and comparing baseline models.
"""
import os.path

import numpy as np
import pandas as pd

from ..utils.utils import *
import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
from captum.attr import FeaturePermutation
import sys
import random
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence
from importlib import import_module, reload
from skopt.space import Real, Integer, Categorical
import torch.utils.data as Data
import time
import json
from copy import deepcopy as cp

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

sys.path.append('configs/')


class Trainer:
    def __init__(self, device='cpu'):
        self.device = device
        self.modelbases = []
        self.modelbases_names = []

    def add_modelbases(self, models: list):
        self.modelbases += models
        self.modelbases_names = [x.program for x in self.modelbases]

    def _get_modelbase(self, program: str):
        if program not in self.modelbases_names:
            raise Exception(f'Program {program} not added to the trainer.')
        return self.modelbases[self.modelbases_names.index(program)]

    def load_config(self, default_configfile: str = None, verbose: bool = True) -> None:
        """
        Load a configfile.
        :param default_configfile: The path to a configfile. If in notebook environment, this parameter is required.
        If the parameter is assigned, the script will not take any input argument from command line. Default to None.
        :param verbose: Whether to output the loaded configs. Default to True.
        :return: None
        """
        base_config = import_module('base_config').BaseConfig().data

        # The base config is loaded using the --base argument
        if is_notebook() and default_configfile is None:
            raise Exception('A config file must be assigned in notebook environment.')
        elif is_notebook() or default_configfile is not None:
            parse_res = {'base': default_configfile}
        else:  # not notebook and configfile is None
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument('--base', required=True)
            for key in base_config.keys():
                if type(base_config[key]) in [str, int, float]:
                    parser.add_argument(f'--{key}', type=type(base_config[key]), required=False)
                elif type(base_config[key]) == list:
                    parser.add_argument(f'--{key}', nargs='+', required=False)
                elif type(base_config[key]) == bool:
                    parser.add_argument(f'--{key}', dest=key, action='store_true')
                    parser.add_argument(f'--no-{key}', dest=key, action='store_false')
                    parser.set_defaults(**{key: base_config[key]})
            parse_res = parser.parse_args().__dict__

        self.configfile = parse_res['base']

        if self.configfile not in sys.modules:
            arg_loaded = import_module(self.configfile).config().data
        else:
            arg_loaded = reload(sys.modules.get(self.configfile)).config().data

        # Then, several args can be modified using other arguments like --lr, --weight_decay
        # only when a config file is not given so that configs depend on input arguments.
        if not is_notebook() and default_configfile is None:
            for key, value in parse_res.items():
                if value is not None:
                    arg_loaded[key] = value

        # Preprocess configs
        tmp_static_params = {}
        for key in arg_loaded['static_params']:
            tmp_static_params[key] = arg_loaded[key]
        arg_loaded['static_params'] = tmp_static_params

        tmp_chosen_params = {}
        for key in arg_loaded['chosen_params']:
            tmp_chosen_params[key] = arg_loaded[key]
        arg_loaded['chosen_params'] = tmp_chosen_params

        key_chosen = list(arg_loaded['chosen_params'].keys())
        key_space = list(arg_loaded['SPACEs'].keys())
        for a, b in zip(key_chosen, key_space):
            if a != b:
                raise Exception('Variables in \'chosen_params\' and \'SPACEs\' should be in the same order.')

        if verbose:
            print(pretty(arg_loaded))

        self.args = arg_loaded

        self.split_by = self.args['split_by']  # 'random' or 'material'

        self.loss = self.args['loss']
        self.bayes_opt = self.args['bayes_opt']

        self.project = self.args['project']
        self.model_name = self.args['model']

        self.static_params = self.args['static_params']
        self.chosen_params = self.args['chosen_params']
        self.layers = self.args['layers']
        self.n_calls = self.args['n_calls']

        SPACE = []
        for var in key_space:
            setting = arg_loaded['SPACEs'][var]
            ty = setting['type']
            setting.pop('type')
            if ty == 'Real':
                SPACE.append(Real(name=var, **setting))
            elif ty == 'Categorical':
                SPACE.append(Categorical(name=var, **setting))
            elif ty == 'Integer':
                SPACE.append(Integer(name=var, **setting))
            else:
                raise Exception('Invalid type of skopt space.')
        self.SPACE = SPACE

        if self.loss == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.loss == 'r2':
            self.loss_fn = r2_loss
        elif self.loss == 'mae':
            self.loss_fn = nn.L1Loss()
        else:
            raise Exception(f'Loss function {self.loss} not implemented.')

        self.params = self.chosen_params

        self.bayes_epoch = self.args['bayes_epoch']

        from src.core.dataprocessor import get_data_processor
        if 'UnscaledDataRecorder' not in self.args['data_processors']:
            if verbose:
                print('UnscaledDataRecorder not in the data_processors pipeline. Only scaled data will be recorded.')
            self.args.append('UnscaledDataRecorder')
        self.dataprocessors = [get_data_processor(name) for name in self.args['data_processors']]

        from src.core.dataderiver import get_data_deriver
        self.dataderivers = [(get_data_deriver(name), kwargs) for name, kwargs in self.args['data_derivers'].items()]

        folder_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '_' + self.configfile

        self.data_path = f'data/{self.project}.xlsx'
        self.project_root = f'output/{self.project}/{folder_name}/'

        if not os.path.exists(f'output/{self.project}'):
            os.mkdir(f'output/{self.project}')
        if not os.path.exists(self.project_root):
            os.mkdir(self.project_root)

        json.dump(arg_loaded, open(self.project_root + 'args.json', 'w'), indent=4)

    def load_data(self, data_path: str = None) -> None:
        """
        Load the data file in ../data directory specified by the 'project' argument in configfile. Data will be splitted
         into training, validation, and testing datasets.
        :param data_path: specify a data file different from 'project'. Default to None.
        :return: None
        """
        if data_path is None:
            self.df = pd.read_excel(self.data_path, engine='openpyxl')
        else:
            self.df = pd.read_excel(data_path, engine='openpyxl')
            self.data_path = data_path

        self.feature_names = list(self.args['feature_names_type'].keys())
        self.label_name = self.args['label_name']

        self.derived_data = {}
        self.derived_data_col_names = {}
        self.derivation_related_cols = []
        self.stacked_derivation_related_cols = []
        for deriver, kwargs in self.dataderivers:
            try:
                value, name, col_names, stacked, related_columns = deriver.derive(self.df, **kwargs)
            except Exception as e:
                print(f'Skip deriver {deriver.__class__.__name__} because of the following exception:')
                print(f'\t{e}')
                continue
            if not stacked:
                self.derived_data[name] = value
                self.derived_data_col_names[name] = col_names
                self.derivation_related_cols += related_columns
            else:
                self.feature_names += col_names
                self.df = pd.concat([self.df, pd.DataFrame(data=value, columns=col_names)], axis=1)
                self.stacked_derivation_related_cols += related_columns

        self.df = self.df.copy().dropna(axis=0, subset=self.label_name + self.derivation_related_cols)
        self._data_process()
        self._rederive_unstacked()
        self._update_dataset_auto()

        print("Dataset size:", len(self.train_dataset), len(self.val_dataset), len(self.test_dataset))

    def _rederive_unstacked(self):
        for deriver, kwargs in self.dataderivers:
            if kwargs['derived_name'] in self.derived_data.keys():
                value, name, col_names, _, _ = deriver.derive(self.df, **kwargs)
                self.derived_data[name] = value
                self.derived_data_col_names[name] = col_names
                if len(self.derived_data[name]) == 0:
                    self.derived_data_col_names.pop(name, None)
                    self.derived_data.pop(name, None)

    def _data_process(self, train_indices=None, val_indices=None, test_indices=None, preprocess=True):
        self.feature_data = self.df[self.feature_names]
        self.label_data = self.df[self.label_name]
        self.unscaled_feature_data = cp(self.feature_data)
        self.unscaled_label_data = cp(self.label_data)
        data, feature_names, label_name = self._get_tabular_dataset()
        data.reset_index(drop=True, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        original_length = len(data)

        if train_indices is None or val_indices is None or test_indices is None:
            train_val_test = np.array([0.6, 0.2, 0.2])
            if self.split_by == "random":
                self.train_indices, self.val_indices, self.test_indices = split_by_random(
                    len(data), train_val_test
                )
            elif self.split_by == "material":
                mat_lay = [str(x) for x in self.df["Material_Code"].copy()]
                mat_lay_set = list(sorted(set(mat_lay)))

                self.train_indices, self.val_indices, self.test_indices = split_by_material(
                    mat_lay, mat_lay_set, train_val_test
                )
            else:
                raise Exception("Split type not implemented")

        if preprocess:
            data = self._data_preprocess(data)

        # Reset indices
        self.retained_indices = np.array(data.index)
        self.dropped_indices = np.setdiff1d(np.arange(original_length), self.retained_indices)
        self.train_indices = np.array(
            [x - np.count_nonzero(self.dropped_indices < x) for x in self.train_indices if x in data.index])
        self.val_indices = np.array(
            [x - np.count_nonzero(self.dropped_indices < x) for x in self.val_indices if x in data.index])
        self.test_indices = np.array(
            [x - np.count_nonzero(self.dropped_indices < x) for x in self.test_indices if x in data.index])

        self.df = pd.DataFrame(self.df.loc[self.retained_indices, :]).reset_index(drop=True)
        self._material_code = pd.DataFrame(self.df['Material_Code'])

        # feature_data and label_data does not contain derived data.
        self.feature_data, self.label_data = self._divide_from_tabular_dataset(data)

    def _data_preprocess(self, input_data: pd.DataFrame):
        data = input_data.copy()
        for processor in self.dataprocessors:
            data = processor.fit_transform(data, self)
        return data

    def _data_transform(self, input_data: pd.DataFrame):
        data = input_data.copy()
        from src.core.dataprocessor import AbstractTransformer
        for processor in self.dataprocessors:
            if issubclass(type(processor), AbstractTransformer):
                data = processor.transform(data, self)
        return data

    def _update_dataset_auto(self):
        self._update_dataset(self.feature_data, self.label_data, self.derived_data.values(),
                                     self.train_indices, self.val_indices, self.test_indices)

    def _update_dataset(self, feature_data, label_data, additional_data, train_indices, val_indices, test_indices):
        X = torch.tensor(feature_data.values.astype(np.float32), dtype=torch.float32).to(
            self.device
        )
        y = torch.tensor(label_data.values.astype(np.float32), dtype=torch.float32).to(
            self.device
        )

        D = [torch.tensor(value, dtype=torch.float32).to(self.device) for value in additional_data]
        dataset = Data.TensorDataset(X, *D, y)

        self.train_dataset = Subset(dataset, train_indices)
        self.val_dataset = Subset(dataset, val_indices)
        self.test_dataset = Subset(dataset, test_indices)
        self.tensors = (X, *D, y) if len(D) > 0 else (X, None, y)

    def describe(self, transformed=False, save=True):
        tabular = self._get_tabular_dataset(transformed=transformed)[0]
        desc = tabular.describe()

        skew = tabular.skew()
        desc = pd.concat(
            [desc, pd.DataFrame(data=skew.values.reshape(len(skew), 1).T, columns=skew.index, index=['Skewness'])],
            axis=0)

        g = self._get_gini(tabular)
        desc = pd.concat([desc, g], axis=0)

        mode, cnt_mode, mode_percent = self._get_mode(tabular)
        desc = pd.concat([desc, mode, cnt_mode, mode_percent], axis=0)

        if save:
            desc.to_csv(self.project_root + 'describe.csv')
        return desc

    def train(self, programs: list = None, verbose: bool = False, debug_mode: bool = False):
        if programs is None:
            modelbases_to_train = self.modelbases
        else:
            modelbases_to_train = [self._get_modelbase(x) for x in programs]

        from src.core.model import TorchModel
        for modelbase in modelbases_to_train:
            if issubclass(type(modelbase), TorchModel) and self.bayes_opt:
                self.params = modelbase._bayes()
            modelbase._train(verbose=verbose, debug_mode=debug_mode)

    def _get_derived_data_sizes(self):
        return [x.shape for x in self.derived_data.values()]

    @staticmethod
    def _get_gini(tabular):
        return pd.DataFrame(data=np.array([[gini(tabular[x]) for x in tabular.columns]]), columns=tabular.columns,
                            index=['Gini Index'])

    @staticmethod
    def _get_mode(tabular):
        mode = tabular.mode().loc[0, :]
        cnt_mode = pd.DataFrame(
            data=np.array([[tabular[mode.index[x]].value_counts()[mode.values[x]] for x in range(len(mode))]]),
            columns=tabular.columns, index=['Mode counts'])
        mode_percent = cnt_mode / tabular.count()
        mode_percent.index = ['Mode percentage']

        mode = pd.DataFrame(data=mode.values.reshape(len(mode), 1).T, columns=mode.index, index=['Mode'])
        return mode, cnt_mode, mode_percent

    def _divide_from_tabular_dataset(self, data: pd.DataFrame):
        feature_data = data[self.feature_names].reset_index(drop=True)
        label_data = data[self.label_name].reset_index(drop=True)

        return feature_data, label_data

    def _get_tabular_dataset(self, transformed=False) -> (pd.DataFrame, list, list):
        if transformed:
            feature_data = self.feature_data
        else:
            feature_data = self.unscaled_feature_data

        feature_names = cp(self.feature_names)
        label_name = cp(self.label_name)

        tabular_dataset = pd.concat([feature_data, self.label_data], axis=1)

        return tabular_dataset, feature_names, label_name

    def get_leaderboard(self, test_data_only: bool = True, dump_trainer=True) -> pd.DataFrame:
        """
        Run all baseline models and the model in this work for a leaderboard.
        :param test_data_only: False to get metrics on training and validation datasets. Default to True.
        :return: The leaderboard dataframe.
        """
        dfs = []
        metrics = ['rmse', 'mse', 'mae', 'mape', 'r2']

        for modelbase in self.modelbases:
            print(f'{modelbase.program} metrics')
            predictions = modelbase._predict_all(verbose=False, test_data_only=test_data_only)
            df = Trainer._metrics(predictions, metrics, test_data_only=test_data_only)
            df['Program'] = modelbase.program
            dfs.append(df)

        df_leaderboard = pd.concat(dfs, axis=0, ignore_index=True)
        df_leaderboard.sort_values('Testing RMSE' if not test_data_only else 'RMSE', inplace=True)
        df_leaderboard.reset_index(drop=True, inplace=True)
        df_leaderboard = df_leaderboard[['Program'] + list(df_leaderboard.columns)[:-1]]
        df_leaderboard.to_csv(self.project_root + 'leaderboard.csv')
        self.leaderboard = df_leaderboard
        if dump_trainer:
            save_trainer(self)
        return df_leaderboard

    def plot_loss(self, train_ls, val_ls):
        """
        Plot loss-epoch while training.
        :return: None
        """
        plt.figure()
        plt.rcParams['font.size'] = 20
        ax = plt.subplot(111)
        ax.plot(
            np.arange(len(train_ls)),
            train_ls,
            label="Training loss",
            linewidth=2,
            color=clr[0],
        )
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
        ax.set_ylabel(f'{self.loss.upper()} Loss')
        plt.savefig(self.project_root + 'loss_epoch.pdf')
        if is_notebook():
            plt.show()
        plt.close()

    def plot_truth_pred(self, program: str = 'ThisWork', log_trans: bool = True, upper_lim=9):
        """
        Comparing ground truth and prediction for different models.
        :param program: Choose a program from 'autogluon', 'pytorch_tabular', 'TabNet'.
        :param log_trans: Whether the target is log10-transformed. Default to True.
        :param upper_lim: The upper boundary of the plot. Default to 9.
        :return: None
        """
        modelbase = self._get_modelbase(program)
        model_names = modelbase._get_model_names()
        predictions = modelbase._predict_all()

        for idx, model_name in enumerate(model_names):
            print(model_name, f'{idx + 1}/{len(model_names)}')
            plt.figure()
            plt.rcParams['font.size'] = 14
            ax = plt.subplot(111)

            self._plot_truth_pred(predictions, ax, model_name, 'Training', clr[0], log_trans=log_trans)
            if 'Validation' in predictions[model_name].keys():
                self._plot_truth_pred(predictions, ax, model_name, 'Validation', clr[2], log_trans=log_trans)
            self._plot_truth_pred(predictions, ax, model_name, 'Testing', clr[1], log_trans=log_trans)

            set_truth_pred(ax, log_trans, upper_lim=upper_lim)

            plt.legend(loc='upper left', markerscale=1.5, handlelength=0.2, handleheight=0.9)

            s = model_name.replace('/', '_')

            plt.savefig(self.project_root + f'{program}/{s}_truth_pred.pdf')
            if is_notebook():
                plt.show()

            plt.close()

    def plot_feature_importance(self, modelbase):
        """
        Calculate and plot permutation feature importance.
        :return: None
        """
        from src.core.model import TorchModel
        if not issubclass(type(modelbase), TorchModel):
            raise Exception('A TorchModel should be passed.')

        def forward_func(data):
            prediction, ground_truth, loss = test_tensor(data,
                                                         self._get_additional_tensors_slice(self.test_dataset.indices),
                                                         self.tensors[-1][self.test_dataset.indices, :],
                                                         modelbase.model,
                                                         self.loss_fn)
            return loss

        feature_perm = FeaturePermutation(forward_func)
        attr = feature_perm.attribute(self.tensors[0][self.test_dataset.indices, :]).cpu().numpy()[0]

        clr = sns.color_palette('deep')

        # if feature type is not assigned in config files, the feature is from dataderiver.
        pal = [clr[self.args['feature_names_type'][x]] if x in self.args['feature_names_type'].keys() else clr[
            len(self.args['feature_types']) - 1] for x in self.feature_names]

        clr_map = dict()
        for idx, feature_type in enumerate(self.args['feature_types']):
            clr_map[feature_type] = clr[idx]

        plt.figure(figsize=(7, 4))
        ax = plt.subplot(111)
        plot_importance(ax, self.feature_names, attr, pal=pal, clr_map=clr_map, linewidth=1, edgecolor='k', orient='h')
        plt.tight_layout()

        plt.savefig(self.project_root + 'feature_importance.png', dpi=600)
        if is_notebook():
            plt.show()
        plt.close()

    def plot_partial_dependence(self, modelbase, log_trans: bool = True, lower_lim=2, upper_lim=7):
        """
        Calculate and plot partial dependence plots.
        :param log_trans: Whether the target is log10-transformed. Default to True.
        :param lower_lim: Lower limit of y-axis when plotting.
        :param upper_lim: Upper limit of y-axis when plotting.
        :return: None
        """
        from src.core.model import TorchModel
        if not issubclass(type(modelbase), TorchModel):
            raise Exception('A TorchModel should be passed.')

        x_values_list = []
        mean_pdp_list = []

        for feature_idx in range(len(self.feature_names)):
            print('Calculate PDP: ', self.feature_names[feature_idx])

            x_value, model_predictions = calculate_pdp(modelbase.model, self.tensors[0][self.train_dataset.indices, :],
                                                       self._get_additional_tensors_slice(self.train_dataset.indices),
                                                       feature_idx,
                                                       grid_size=30)

            x_values_list.append(x_value)
            mean_pdp_list.append(model_predictions)

        fig = plot_pdp(self.feature_names, x_values_list, mean_pdp_list, self.tensors[0], self.train_dataset.indices,
                       log_trans=log_trans, lower_lim=lower_lim, upper_lim=upper_lim)

        plt.savefig(self.project_root + 'partial_dependence.pdf')
        if is_notebook():
            plt.show()
        plt.close()

    def plot_partial_err(self, modelbase, thres=0.8):
        """
        Calculate and plot partial error dependency for each feature.
        :param thres: Points with loss higher than thres will be marked.
        :return: None
        """
        from src.core.model import TorchModel
        if not issubclass(type(modelbase), TorchModel):
            raise Exception('A TorchModel should be passed.')
        prediction, ground_truth, loss = test_tensor(self.tensors[0][self.test_dataset.indices, :],
                                                     self._get_additional_tensors_slice(self.test_dataset.indices),
                                                     self.tensors[-1][self.test_dataset.indices, :], modelbase.model,
                                                     self.loss_fn)
        plot_partial_err(self.feature_data.loc[np.array(self.test_dataset.indices), :].reset_index(drop=True),
                         ground_truth, prediction, thres=thres)

        plt.savefig(self.project_root + 'partial_err.pdf')
        if is_notebook():
            plt.show()
        plt.close()

    def plot_corr(self, fontsize=10, cmap='bwr'):
        """
        Plot Pearson correlation among features and the target.
        :return: None
        """
        feature_names = self.feature_names + self.label_name
        # sns.reset_defaults()
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(111)
        plt.box(on=True)
        df_all = pd.concat([self.feature_data, self.label_data], axis=1)
        corr = df_all.corr().values
        im = ax.imshow(corr, cmap=cmap)
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_yticks(np.arange(len(feature_names)))

        ax.set_xticklabels(feature_names, fontsize=fontsize)
        ax.set_yticklabels(feature_names, fontsize=fontsize)

        plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
                 rotation_mode="anchor")

        norm_corr = corr - (np.max(corr) + np.min(corr)) / 2
        norm_corr /= np.max(norm_corr)

        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                text = ax.text(j, i, round(corr[i, j], 2),
                               ha="center", va="center", color="w" if np.abs(norm_corr[i, j]) > 0.3 else 'k',
                               fontsize=fontsize)

        plt.tight_layout()
        plt.savefig(self.project_root + 'corr.pdf')
        if is_notebook():
            plt.show()
        plt.close()

    def plot_pairplot(self, **kwargs):
        df_all = pd.concat([self.unscaled_feature_data, self.unscaled_label_data], axis=1)
        sns.pairplot(df_all, corner=True, diag_kind='kde', **kwargs)
        plt.tight_layout()
        plt.savefig(self.project_root + 'pair.jpg')
        if is_notebook():
            plt.show()
        plt.close()

    def plot_feature_box(self):
        # sns.reset_defaults()
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111)
        bp = sns.boxplot(
            data=self.feature_data,
            orient='h', linewidth=1,
            fliersize=4, flierprops={'marker': 'o'})

        boxes = []

        for x in ax.get_children():
            if isinstance(x, matplotlib.patches.PathPatch):
                boxes.append(x)

        color = '#639FFF'

        for patch in boxes:
            patch.set_facecolor(color)

        plt.grid(linewidth=0.4, axis='x')
        ax.set_axisbelow(True)
        plt.ylabel('Values (Standard Scaled)')
        # ax.tick_params(axis='x', rotation=90)
        plt.tight_layout()
        plt.savefig(self.project_root + 'feature_box.pdf')
        plt.show()
        plt.close()

    def _get_indices(self, partition='train'):
        indices_map = {
            'train': self.train_indices,
            'val': self.val_indices,
            'test': self.test_indices,
            'all': np.array(self.feature_data.index)
        }

        if partition not in indices_map.keys():
            raise Exception(f'Partition {train} not available. Select among {list(indices_map.keys())}')

        return indices_map[partition]

    def get_material_code(self, unique=False, partition='all'):
        indices = self._get_indices(partition=partition)
        if unique:
            unique_list = list(sorted(set(self._material_code.loc[indices, 'Material_Code'])))
            val_cnt = self._material_code.loc[indices, :].value_counts()
            return pd.DataFrame({
                'Material_Code': unique_list,
                'Count': [val_cnt[x].values[0] for x in unique_list]})
        else:
            return self._material_code.loc[indices, :]

    def _select_by_material_code(self, m_code: str, partition='all'):
        code_df = self.get_material_code(unique=False, partition=partition)
        if m_code not in code_df['Material_Code'].values:
            raise Exception(f'Material code {m_code} not available.')
        return code_df.index[np.where(code_df['Material_Code'] == m_code)[0]]

    def plot_S_N(self, s_col, n_col, m_code, load_dir='tension', ax=None, grid_size=100):
        if s_col not in self.df.columns:
            raise Exception(f'{s_col} not in features.')
        if n_col not in self.label_name:
            raise Exception(f'{n_col} is not the target.')
        m_train_indices = self._select_by_material_code(m_code, partition='train')
        m_test_indices = self._select_by_material_code(m_code, partition='test')
        m_val_indices = self._select_by_material_code(m_code, partition='val')
        sgn = 1 if load_dir == 'tension' else -1
        m_train_indices = m_train_indices[self.df.loc[m_train_indices, s_col].values * sgn > 0]
        m_test_indices = m_test_indices[self.df.loc[m_test_indices, s_col].values * sgn > 0]
        m_val_indices = m_val_indices[self.df.loc[m_val_indices, s_col].values * sgn > 0]

        s_train = self.df.loc[m_train_indices, s_col]
        n_train = self.df.loc[m_train_indices, n_col]
        s_val = self.df.loc[m_val_indices, s_col]
        n_val = self.df.loc[m_val_indices, n_col]
        s_test = self.df.loc[m_test_indices, s_col]
        n_test = self.df.loc[m_test_indices, n_col]

        all_unscaled_s = np.vstack(
            [s_train.values.reshape(-1, 1), s_val.values.reshape(-1, 1), s_test.values.reshape(-1, 1)])
        s_unscaled_perm = np.linspace(np.min(all_unscaled_s), np.max(all_unscaled_s), grid_size)
        df_perm = pd.DataFrame(data=np.repeat(self.df.loc[m_train_indices[0], :].values.reshape(-1, 1),
                                              repeats=grid_size, axis=1).T,
                               columns=self.df.columns, index=[x for x in range(grid_size)])
        df_perm[s_col] = s_unscaled_perm
        if s_col in self.stacked_derivation_related_cols + self.derivation_related_cols:
            additional_data = []
            for deriver, kwargs in self.dataderivers:
                try:
                    value, name, col_names, stacked, related_columns = deriver.derive(df_perm, **kwargs)
                except Exception as e:
                    continue
                if stacked:
                    df_perm[col_names] = value
                else:
                    additional_data.append(value)
        else:
            additional_data = list(self.derived_data.values())
        df_perm = self._data_transform(df_perm)

        best_model_program, best_model_name = self._get_best_model()
        pred_n_perm = self._get_modelbase(best_model_program)._predict(df_perm, additional_data=additional_data,
                                                                       model_name=best_model_name)

        if ax is None:
            new_ax = True
            plt.figure()
            plt.rcParams['font.size'] = 14
            ax = plt.subplot(111)
        else:
            new_ax = False

        def scatter_func(x, y, color, name):
            ax.scatter(x, y,
                       s=20,
                       color=color,
                       marker='o',
                       label=f"{name} dataset",
                       linewidth=0.4,
                       edgecolors="k")

        scatter_func(n_train, s_train, clr[0], 'Training')
        scatter_func(n_val, s_val, clr[1], 'Validation')
        scatter_func(n_test, s_test, clr[2], 'Testing')

        ax.plot(pred_n_perm, s_unscaled_perm)

        ax.legend(loc='upper right', markerscale=1.5, handlelength=0.2, handleheight=0.9)
        ax.set_xlabel(n_col)
        ax.set_ylabel(s_col)

        if is_notebook() and new_ax:
            plt.show()
        if new_ax:
            plt.close()

    def _get_best_model(self):
        if not hasattr(self, 'leaderboard'):
            self.get_leaderboard(test_data_only=True, dump_trainer=False)
        return self.leaderboard['Program'].values[0], self.leaderboard['Model'].values[0]

    @staticmethod
    def _metrics(predictions, metrics, test_data_only):
        df_metrics = pd.DataFrame()
        for model_name, model_predictions in predictions.items():
            df = pd.DataFrame(index=[0])
            df['Model'] = model_name
            for tvt, (y_pred, y_true) in model_predictions.items():
                if test_data_only and tvt != 'Testing':
                    continue
                for metric in metrics:
                    metric_value = Trainer._metric_sklearn(y_true, y_pred, metric)
                    df[tvt + ' ' + metric.upper() if not test_data_only else metric.upper()] = metric_value
            df_metrics = pd.concat([df_metrics, df], axis=0, ignore_index=True)

        return df_metrics

    @staticmethod
    def _metric_sklearn(y_true, y_pred, metric):
        if metric == 'mse':
            from sklearn.metrics import mean_squared_error
            return mean_squared_error(y_true, y_pred)
        elif metric == 'rmse':
            from sklearn.metrics import mean_squared_error
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == 'mae':
            from sklearn.metrics import mean_absolute_error
            return mean_absolute_error(y_true, y_pred)
        elif metric == 'mape':
            from sklearn.metrics import mean_absolute_percentage_error
            return mean_absolute_percentage_error(y_true, y_pred)
        elif metric == 'r2':
            from sklearn.metrics import r2_score
            return r2_score(y_true, y_pred)
        else:
            raise Exception(f'Metric {metric} not implemented.')

    def _plot_truth_pred(self, predictions, ax, model_name, name, color, marker='o', log_trans=True, verbose=True):
        pred_y, y = predictions[model_name][name]
        r2 = Trainer._metric_sklearn(y, pred_y, 'r2')
        loss = self.loss_fn(torch.Tensor(y), torch.Tensor(pred_y))
        if verbose:
            print(f"{name} Loss: {loss:.4f}, R2: {r2:.4f}")
        ax.scatter(10 ** y if log_trans else y, 10 ** pred_y if log_trans else pred_y,
                   s=20,
                   color=color,
                   marker=marker,
                   label=f"{name} dataset ($R^2$={r2:.3f})",
                   linewidth=0.4,
                   edgecolors="k")
        ax.set_xlabel("Ground truth")
        ax.set_ylabel("Prediction")

    def _get_additional_tensors_slice(self, indices):
        res = []
        for tensor in self.tensors[1:len(self.tensors) - 1]:
            if tensor is not None:
                res.append(tensor[indices, :])
        return tuple(res)


def save_trainer(trainer, path=None, verbose=True):
    import pickle
    path = trainer.project_root + 'trainer.pkl' if path is None else path
    if verbose:
        print(f'Trainer saved. To load the trainer, run trainer = load_trainer(path=\'{path}\')')
    with open(path, 'wb') as outp:
        pickle.dump(trainer, outp, pickle.HIGHEST_PROTOCOL)


def load_trainer(path=None):
    import pickle
    with open(path, 'rb') as inp:
        trainer = pickle.load(inp)
    return trainer
