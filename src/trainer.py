"""
The basic class for the project. It includes configuration, data processing, training, plotting,
and comparing with baseline models.
"""
import os.path

import numpy as np

from utils import *
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from captum.attr import FeaturePermutation
import sys
import random
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence
from importlib import import_module, reload
from skopt.space import Real, Integer, Categorical

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

sys.path.append('../configs/')


class Trainer():
    def __init__(self, device):
        self.device = device

    def load_config(self, default_configfile, verbose=False):
        if is_notebook():
            self.configfile = default_configfile
        else:
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument('--configfile', default=default_configfile)
            self.configfile = parser.parse_args().configfile

        if self.configfile not in sys.modules:
            arg_loaded = import_module(self.configfile).config().data
        else:
            arg_loaded = reload(sys.modules.get(self.configfile)).config().data

        if verbose:
            print(arg_loaded)

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

        SPACE = []
        for var in key_space:
            setting = arg_loaded['SPACEs'][var]
            type = setting['type']
            setting.pop('type')
            if type == 'Real':
                SPACE.append(Real(name=var, **setting))
            elif type == 'Categorical':
                SPACE.append(Categorical(name=var, **setting))
            elif type == 'Integer':
                SPACE.append(Integer(name=var, **setting))
            else:
                raise Exception('Invalid type of skopt space.')
        arg_loaded['SPACE'] = SPACE

        self.args = arg_loaded

        self.split_by = self.args['split_by']  # 'random' or 'material'

        self.validation = self.args['validation']
        self.loss = self.args['loss']
        self.bayes_opt = self.args['bayes_opt']

        self.project = self.args['project']
        self.model_name = self.args['model']

        self.data_path = f'../data/{self.project}_fatigue.xlsx'
        self.ckp_path = f'../output/{self.project}/fatigue.pt'
        self.skopt_path = f'../output/{self.project}/skopt.pt'

        self.static_params = self.args['static_params']
        self.chosen_params = self.args['chosen_params']
        self.layers = self.args['layers']
        self.n_calls = self.args['n_calls']
        self.SPACE = self.args['SPACE']

        if self.loss == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.loss == 'r2':
            self.loss_fn = r2_loss
        elif self.loss == 'rmse':
            self.loss_fn = RMSELoss()
        elif self.loss == 'mae':
            self.loss_fn = nn.L1Loss()
        else:
            raise Exception(f'Loss function {self.loss} not implemented.')

        self.params = self.chosen_params

        self.use_sequence = self.args['sequence']

        if not os.path.exists(f'../output/{self.project}'):
            os.mkdir(f'../output/{self.project}')

    def load_data(self, data_path=None, impute=False):
        if data_path is None:
            self.df = pd.read_excel(self.data_path, engine='openpyxl')
        else:
            self.df = pd.read_excel(data_path, engine='openpyxl')
            self.data_path = data_path

        self.feature_names = list(self.args['feature_names_type'].keys())

        self.label_name = self.args['label_name']

        if self.use_sequence and 'Sequence' in self.df.columns:
            self.sequence = [[int(y) if y != 'nan' else np.nan for y in str(x).split('/')] for x in
                             self.df['Sequence'].values]

            self.deg_layers = np.zeros((len(self.sequence), 4),
                                       dtype=np.int)  # for 0-deg, pm45-deg, 90-deg, and other directions respectively

            for idx, seq in enumerate(self.sequence):
                self.deg_layers[idx, 0] = seq.count(0)
                self.deg_layers[idx, 1] = seq.count(45) + seq.count(-45)
                self.deg_layers[idx, 2] = seq.count(90)
                self.deg_layers[idx, 3] = len(seq) - seq.count(np.nan) - np.sum(self.deg_layers[idx, :3])
        elif self.use_sequence and 'Sequence' not in self.df.columns:
            print('No sequence infomation in the dataframe. use_sequence off.')
            self.use_sequence = False
            self.deg_layers = None
        else:
            self.deg_layers = None

        self.feature_data, self.label_data, self.tensors, \
        self.train_dataset, self.val_dataset, self.test_dataset, self.scaler = split_dataset(
            self.df,
            self.deg_layers,
            self.feature_names,
            self.label_name,
            self.device,
            self.validation,
            self.split_by,
            impute=impute
        )

    def bayes(self):
        if not self.bayes_opt:
            print('Bayes optimization not activated in configuration file. Return preset chosen_params.')
            return self.chosen_params

        # If def is not global, pickle will raise 'Can't get local attribute ...'
        # IT IS NOT SAFE, BUT I DID NOT FIND A BETTER SOLUTION
        global _trainer_bayes_objective, _trainer_bayes_callback

        bar = tqdm(total=self.n_calls)

        @skopt.utils.use_named_args(self.SPACE)
        def _trainer_bayes_objective(**params):
            res = model_train(self.train_dataset, self.val_dataset, self.validation,
                              self.loss_fn, self.ckp_path,
                              model=self.new_model(),
                              verbose=False, return_loss_list=False, **{**params, **self.static_params})

            return res

        postfix = {'Current loss': 1e8, 'Minimum': 1e8, 'Params': list(self.chosen_params.values()),
                   'Minimum at call': 0}

        def _trainer_bayes_callback(result):
            postfix['Current loss'] = result.func_vals[-1]

            if result.fun < postfix['Minimum']:
                postfix['Minimum'] = result.fun
                postfix['Params'] = result.x
                postfix['Minimum at call'] = len(result.func_vals)
            skopt.dump(result, self.skopt_path)

            if len(result.func_vals) % 5 == 0:
                plt.figure()
                ax = plt.subplot(111)
                ax = plot_convergence(result, ax)
                plt.savefig(f'../output/{self.project}/skopt_convergence.pdf')
                plt.close()

            bar.set_postfix(**postfix)
            bar.update(1)

        result = gp_minimize(_trainer_bayes_objective, self.SPACE, n_calls=self.n_calls, random_state=0,
                             x0=list(self.chosen_params.values()),
                             callback=_trainer_bayes_callback)
        print(result.func_vals.min())

        params = {}
        for key, value in zip(self.chosen_params.keys(), result.x):
            params[key] = value

        return params

    def train(self, verbose_per_epoch=100):
        self.model = self.new_model()

        min_loss, self.train_ls, self.val_ls = model_train(self.train_dataset, self.val_dataset,
                                                           self.validation, self.loss_fn,
                                                           self.ckp_path, model=self.model,
                                                           verbose_per_epoch=verbose_per_epoch,
                                                           **{**self.params, **self.static_params})

        if self.validation:
            self.model.load_state_dict(torch.load(self.ckp_path))
        else:
            torch.save(self.model.state_dict(), self.ckp_path)

        print('Minimum loss:', min_loss)

    def new_model(self):
        if self.model_name == 'MLP':
            return NN(len(self.feature_names), len(self.label_name), self.layers, self.use_sequence).to(self.device)
        else:
            raise Exception(f'Model {self.model_name} not implemented.')

    def plot_loss(self):
        plt.figure()
        plt.rcParams['font.size'] = 20
        ax = plt.subplot(111)
        plot_loss(self.train_ls, self.val_ls, ax)
        ax.set_ylabel(f'{self.loss.upper()} Loss')
        plt.savefig(f'../output/{self.project}/loss_epoch.pdf')
        if is_notebook():
            plt.show()
        plt.close()

    def plot_truth_pred(self):
        plt.figure()
        plt.rcParams['font.size'] = 14
        ax = plt.subplot(111)
        plot_truth_pred_NN(self.train_dataset, self.val_dataset, self.test_dataset, self.model, self.loss_fn, ax)
        plt.legend(loc='upper left', markerscale=1.5, handlelength=0.2, handleheight=0.9)

        plt.savefig(f'../output/{self.project}/truth_pred.pdf')
        if is_notebook():
            plt.show()
        plt.close()

    def plot_truth_pred_sklearn(self, model_name):
        plt.figure()
        plt.rcParams['font.size'] = 14
        ax = plt.subplot(111)

        if model_name == 'rf':
            if self.split_by == 'material':
                rf = RandomForestRegressor(n_jobs=-1, max_depth=6)
            else:
                rf = RandomForestRegressor(n_jobs=-1, max_depth=15)

            plot_truth_pred_sklearn(self.feature_data, self.label_data, self.train_dataset.indices,
                                    self.test_dataset.indices, ax, model=rf,
                                    split_by=self.split_by)
        elif model_name == 'svm':
            sv = svm.SVR()
            plot_truth_pred_sklearn(self.feature_data, self.label_data, self.train_dataset.indices,
                                    self.test_dataset.indices, ax, model=sv,
                                    split_by=self.split_by)
        else:
            plt.close()
            raise Exception('Sklearn model not implemented')

        plt.legend(loc='upper left', markerscale=1.5, handlelength=0.2, handleheight=0.9)

        plt.savefig(f'../output/{self.project}/{model_name}_truth_pred.pdf')
        if is_notebook():
            plt.show()
        plt.close()

    def plot_feature_importance(self):
        def forward_func(data):
            prediction, ground_truth, loss = test_tensor(data, self._get_additional_tensors_slice(self.test_dataset.indices), self.tensors[-1][self.test_dataset.indices, :], self.model,
                                                         self.loss_fn)
            return loss

        feature_perm = FeaturePermutation(forward_func)
        attr = feature_perm.attribute(self.tensors[0][self.test_dataset.indices, :]).cpu().numpy()[0]

        clr = sns.color_palette('deep')

        pal = [clr[self.args['feature_names_type'][x]] for x in self.feature_names]

        clr_map = dict()
        for idx, feature_type in enumerate(self.args['feature_types']):
            clr_map[feature_type] = clr[idx]

        plt.figure(figsize=(7, 4))
        ax = plt.subplot(111)
        plot_importance(ax, self.feature_names, attr, pal=pal, clr_map=clr_map, linewidth=1, edgecolor='k', orient='h')
        plt.tight_layout()

        plt.savefig(f'../output/{self.project}/feature_importance.png', dpi=600)
        if is_notebook():
            plt.show()
        plt.close()

    def plot_partial_dependence(self):
        x_values_list = []
        mean_pdp_list = []

        for feature_idx in range(len(self.feature_names)):
            print('Calculate PDP: ', self.feature_names[feature_idx])

            x_value, model_predictions = calculate_pdp(self.model, self.tensors[0][self.train_dataset.indices, :],
                                                       self._get_additional_tensors_slice(self.train_dataset.indices),
                                                       feature_idx,
                                                       grid_size=30)

            x_values_list.append(x_value)
            mean_pdp_list.append(model_predictions)

        fig = plot_pdp(self.feature_names, x_values_list, mean_pdp_list, self.tensors[0], self.train_dataset.indices)

        plt.savefig(f'../output/{self.project}/partial_dependence.pdf')
        if is_notebook():
            plt.show()
        plt.close()

    def plot_partial_err(self):
        prediction, ground_truth, loss = test_tensor(self.tensors[0][self.test_dataset.indices, :],
                                                     self._get_additional_tensors_slice(self.test_dataset.indices),
                                                     self.tensors[-1][self.test_dataset.indices, :], self.model,
                                                     self.loss_fn)
        plot_partial_err(self.feature_data.loc[np.array(self.test_dataset.indices), :].reset_index(drop=True),
                         ground_truth, prediction)

        plt.savefig(f'../output/{self.project}/partial_err.pdf')
        if is_notebook():
            plt.show()
        plt.close()

    def plot_corr(self):
        fig = plt.figure(figsize=(10,10))
        ax = plt.subplot(111)
        df_all = pd.concat([self.feature_data, self.label_data], axis=1)
        corr = df_all.corr()
        sns.heatmap(corr, ax=ax, annot=True, xticklabels=corr.columns, yticklabels=corr.columns, square=True, cmap='Blues', cbar=False)
        plt.tight_layout()
        plt.savefig(f'../output/{self.project}/corr.pdf')
        if is_notebook():
            plt.show()
        plt.close()

    def _get_additional_tensors_slice(self, indices):
        res = []
        for tensor in self.tensors[1:len(self.tensors)-1]:
            if tensor is not None:
                res.append(tensor[indices, :])
        return tuple(res)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using {} device".format(device))

    configfile = 'pr-FACT_sp-random_va-True_ph-False_ba-False_pa-500_ep-2000_lr-003_we-002_ba-1024_n-200_se-True'

    trainer = Trainer(device=device)
    ## Set params
    trainer.load_config(default_configfile=configfile)
    ## Set datasets
    trainer.load_data()

    trainer.train()
    trainer.plot_loss()