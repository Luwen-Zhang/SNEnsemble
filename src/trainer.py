"""
The basic class for the project. It includes configuration, data processing, training, plotting,
and comparing with baseline models.
"""

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

torch.use_deterministic_algorithms(True)
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

sys.path.append('../configs/')


class Trainer():
    def __init__(self, device):
        self.device = device

    def load_config(self, default_configfile):
        if is_notebook():
            self.configfile = default_configfile
        else:
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument('--configfile', default=default_configfile)
            self.configfile = parser.parse_args().configfile

        if self.configfile not in sys.modules:
            arg_loaded = import_module(self.configfile).config.data
        else:
            arg_loaded = reload(sys.modules.get(self.configfile)).config.data

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
        self.physics_informed = self.args['physics_informed']
        self.bayes_opt = self.args['bayes_opt']

        self.project = self.args['project']

        self.data_path = f'../data/{self.project}_fatigue.xlsx'
        self.ckp_path = f'../output/{self.project}/fatigue.pt'
        self.skopt_path = f'../output/{self.project}/skopt.pt'

        self.static_params = self.args['static_params']
        self.chosen_params = self.args['chosen_params']
        self.layers = self.args['layers']
        self.n_calls = self.args['n_calls']
        self.SPACE = self.args['SPACE']

        if self.physics_informed:
            self.loss_fn = PI_MSELoss()
        else:
            self.loss_fn = nn.MSELoss()

        self.params = self.chosen_params

        self.use_sequence = self.args['sequence']

    def load_data(self, data_path=None):
        if data_path is None:
            self.df = pd.read_excel(self.data_path, engine='openpyxl')
        else:
            self.df = pd.read_excel(data_path, engine='openpyxl')
            self.data_path = data_path

        self.feature_names = list(self.args['feature_names_type'].keys())

        self.label_name = self.args['label_name']

        self.feature_data, self.label_data, self.X, self.y, \
        self.train_dataset, self.val_dataset, self.test_dataset, self.scaler = split_dataset(
            self.df,
            self.feature_names,
            self.label_name,
            self.device,
            self.validation,
            self.split_by)

        if self.use_sequence:
            self._all_sequence = np.array(
                [np.array([int(y) if y != 'nan' else np.nan for y in str(x).split('/')]) for x in
                 self.df['Sequence'].values], dtype=object)
            self.train_sequence = self._all_sequence[self.train_dataset.indices]
            self.val_sequence = self._all_sequence[self.val_dataset.indices] if self.val_dataset is not None else None
            self.test_sequence = self._all_sequence[self.test_dataset.indices]

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
            res = model_train(self.train_dataset, self.test_dataset, self.val_dataset, self.layers, self.validation,
                              self.loss_fn, self.ckp_path,
                              self.device,
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
                plt.savefig(f'../output/{self.project}/skopt_convergence.svg')
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
        self.model = NN(len(self.feature_names), len(self.label_name), self.layers).to(self.device)

        min_loss, self.train_ls, self.val_ls = model_train(self.train_dataset, self.test_dataset, self.val_dataset,
                                                           self.layers, self.validation, self.loss_fn,
                                                           self.ckp_path, self.device, model=self.model,
                                                           verbose_per_epoch=verbose_per_epoch,
                                                           **{**self.params, **self.static_params})

        if self.validation:
            self.model.load_state_dict(torch.load(self.ckp_path))
        else:
            torch.save(self.model.state_dict(), self.ckp_path)

        print('Minimum loss:', min_loss)

    def plot_loss(self):
        plt.figure()
        plt.rcParams['font.size'] = 20
        ax = plt.subplot(111)
        plot_loss(self.train_ls, self.val_ls, ax)
        plt.savefig(f'../output/{self.project}/loss_epoch.svg')
        if is_notebook():
            plt.show()
        plt.close()

    def plot_truth_pred(self):
        plt.figure()
        plt.rcParams['font.size'] = 14
        ax = plt.subplot(111)
        plot_truth_pred_NN(self.train_dataset, self.val_dataset, self.test_dataset, self.model, self.loss_fn, ax)
        plt.legend(loc='upper left', markerscale=1.5, handlelength=0.2, handleheight=0.9)

        plt.savefig(f'../output/{self.project}/truth_pred.svg')
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

        plt.savefig(f'../output/{self.project}/{model_name}_truth_pred.svg')
        if is_notebook():
            plt.show()
        plt.close()

    def plot_feature_importance(self):
        def forward_func(data):
            prediction, ground_truth, loss = test_tensor(data, self.y[self.test_dataset.indices, :], self.model,
                                                         self.loss_fn)
            return loss

        feature_perm = FeaturePermutation(forward_func)
        attr = feature_perm.attribute(self.X[self.test_dataset.indices, :]).cpu().numpy()[0]

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

            x_value, model_predictions = calculate_pdp(self.model, self.X[self.train_dataset.indices, :], feature_idx,
                                                       grid_size=30)

            x_values_list.append(x_value)
            mean_pdp_list.append(model_predictions)

        fig = plot_pdp(self.feature_names, x_values_list, mean_pdp_list, self.X, self.train_dataset.indices)

        plt.savefig(f'../output/{self.project}/partial_dependence.svg')
        if is_notebook():
            plt.show()
        plt.close()

    def plot_partial_err(self):
        prediction, ground_truth, loss = test_tensor(self.X[self.test_dataset.indices, :],
                                                     self.y[self.test_dataset.indices, :], self.model,
                                                     self.loss_fn)
        plot_partial_err(self.feature_data.loc[np.array(self.test_dataset.indices), :].reset_index(drop=True),
                         ground_truth, prediction)

        plt.savefig(f'../output/{self.project}/partial_err.svg')
        if is_notebook():
            plt.show()
        plt.close()
