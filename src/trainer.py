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

    def load_config(self, default_configfile=None, verbose=True):
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
                elif type(base_config[key]) == bool:
                    parser.add_argument(f'--{key}', dest=key, action='store_true')
                    parser.add_argument(f'--no-{key}', dest=key, action='store_false')
            parse_res = parser.parse_args().__dict__

        self.configfile = parse_res['base']

        if self.configfile not in sys.modules:
            arg_loaded = import_module(self.configfile).config().data
        else:
            arg_loaded = reload(sys.modules.get(self.configfile)).config().data

        # Then, several args can be modified using other arguments like --lr, --weight_decay
        # only when a config file is not given so that configs depend on input arguments.
        if not is_notebook() and default_configfile is None:
            for key, value in zip(parse_res.keys(), parse_res.values()):
                if value is not None:
                    arg_loaded[key] = value

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
        arg_loaded['SPACE'] = SPACE

        if verbose:
            print(arg_loaded)

        self.args = arg_loaded

        self.split_by = self.args['split_by']  # 'random' or 'material'

        self.loss = self.args['loss']
        self.bayes_opt = self.args['bayes_opt']

        self.project = self.args['project']
        self.model_name = self.args['model']

        self.data_path = f'../data/{self.project}_fatigue.xlsx'
        self.ckp_path = f'../output/{self.project}/fatigue.pt'
        self.skopt_path = f'../output/{self.project}/skopt.pt'
        self.project_root = f'../output/{self.project}/'

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
            res, _, _ = self._model_train(model=self.new_model(),
                                          verbose=False,
                                          **{**params, **self.static_params})

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

        min_loss, self.train_ls, self.val_ls = self._model_train(model=self.model,
                                                                 verbose_per_epoch=verbose_per_epoch,
                                                                 **{**self.params, **self.static_params})

        self.model.load_state_dict(torch.load(self.ckp_path))

        print('Minimum loss:', min_loss)

    def new_model(self):
        if self.model_name == 'MLP':
            return NN(len(self.feature_names), len(self.label_name), self.layers, self.use_sequence).to(self.device)
        else:
            raise Exception(f'Model {self.model_name} not implemented.')

    def autogluon_tests(self, verbose=False):
        print('\n-------------Run AutoGluon Tests-------------\n')
        # https://github.com/awslabs/autogluon
        self._disable_tqdm()
        import warnings
        warnings.simplefilter(action='ignore', category=UserWarning)
        from autogluon.tabular import TabularPredictor
        tabular_dataset = pd.concat([self.feature_data, self.label_data], axis=1)
        predictor = TabularPredictor(label=self.label_name[0], path=self.project_root+'autogluon')
        with HiddenPrints(disable_logging=True if not verbose else False):
            predictor.fit(tabular_dataset.loc[self.train_dataset.indices + self.val_dataset.indices, :],
                          presets='best_quality', hyperparameter_tune_kwargs='bayesopt', verbosity=0 if not verbose else 2)
        self.autogluon_leaderboard = predictor.leaderboard(tabular_dataset.loc[self.test_dataset.indices, :])
        # y_pred = predictor.predict(tabular_dataset.loc[self.test_dataset.indices, self.feature_names])
        # predictor.evaluate_predictions(y_true=tabular_dataset.loc[self.test_dataset.indices, self.label_name[0]],
        #                                y_pred=y_pred)
        if verbose:
            print(self.autogluon_leaderboard)
        self._enable_tqdm()
        warnings.simplefilter(action='default', category=UserWarning)
        print('\n-------------AutoGluon Tests End-------------\n')

    def pytorch_tabular_tests(self, verbose=False):
        print('\n-------------Run Pytorch-tabular Tests-------------\n')
        # https://github.com/manujosephv/pytorch_tabular
        import warnings
        warnings.simplefilter(action='ignore', category=UserWarning)
        self._disable_tqdm()
        from pytorch_tabular import TabularModel
        from pytorch_tabular.models import CategoryEmbeddingModelConfig, \
            NodeConfig, TabNetModelConfig, TabTransformerConfig, AutoIntConfig
        from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

        data_config = DataConfig(
            target=self.label_name,
            continuous_cols=self.feature_names,
            num_workers=8
        )

        trainer_config = TrainerConfig(
            auto_lr_find=False,
            max_epochs=1000,
            early_stopping_patience=100,
        )
        optimizer_config = OptimizerConfig(
        )

        models = [
            CategoryEmbeddingModelConfig(task='regression'),
            NodeConfig(task='regression'),
            TabNetModelConfig(task='regression'),
            TabTransformerConfig(task='regression'),
            AutoIntConfig(task='regression')
        ]

        ####################################
        # If using PyCharm, see lollows' answer on https://stackoverflow.com/questions/59455268/how-to-disable-progress-bar-in-pytorch-lightning/66731318#66731318
        # to fix progress bar error
        ####################################
        metrics = {'model': [], 'mse': []}
        for model_config in models:
            print('Training', model_config)
            with HiddenPrints(disable_logging=True if not verbose else False):
                tabular_model = TabularModel(
                    data_config=data_config,
                    model_config=model_config,
                    optimizer_config=optimizer_config,
                    trainer_config=trainer_config
                )
                tabular_model.config.checkpoints_path = self.project_root + 'pytorch_tabular'
                tabular_dataset = pd.concat([self.feature_data, self.label_data], axis=1)

                tabular_model.fit(train=tabular_dataset.loc[self.train_dataset.indices,:],
                                  validation=tabular_dataset.loc[self.val_dataset.indices,:],
                                  loss=self.loss_fn)
                # tabular_model.evaluate(tabular_dataset.loc[self.train_dataset.indices, :])
                # y_pred = tabular_model.predict(tabular_dataset.loc[self.test_dataset.indices,:])
                metrics['model'].append(model_config.__class__.__name__.replace('Config', ''))
                metrics['mse'].append(tabular_model.evaluate(tabular_dataset.loc[self.test_dataset.indices,:])[0]['test_mean_squared_error'])

        metrics = pd.DataFrame(metrics)
        metrics['rmse'] = metrics['mse']**(1/2)
        metrics.sort_values('rmse', inplace=True)
        self.pytorch_tabular_leaderboard = metrics
        print(self.pytorch_tabular_leaderboard)

        self._enable_tqdm()
        warnings.simplefilter(action='default', category=UserWarning)
        print('\n-------------Pytorch-tabular Tests End-------------\n')

    def plot_loss(self):
        plt.figure()
        plt.rcParams['font.size'] = 20
        ax = plt.subplot(111)
        ax.plot(
            np.arange(len(self.train_ls)),
            self.train_ls,
            label="Train loss",
            linewidth=2,
            color=clr[0],
        )
        if len(self.val_ls) > 0:
            ax.plot(
                np.arange(len(self.val_ls)),
                self.val_ls,
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

        train_indices = self.train_dataset.indices
        val_indices = self.val_dataset.indices if self.val_dataset is not None else None
        test_indices = self.test_dataset.indices

        train_x = self.feature_data.values[np.array(train_indices), :]
        test_x = self.feature_data.values[np.array(test_indices), :]

        train_y = self.label_data.values[np.array(train_indices), :].reshape(-1, 1)
        test_y = self.label_data.values[np.array(test_indices), :].reshape(-1, 1)

        if self.use_sequence:
            train_x = np.hstack([train_x, self.deg_layers[np.array(train_indices), :]])
            test_x = np.hstack([test_x, self.deg_layers[np.array(test_indices), :]])

        if val_indices is not None:
            val_x = self.feature_data.values[np.array(val_indices), :]
            val_y = self.label_data.values[np.array(val_indices), :].reshape(-1, 1)

            if self.use_sequence:
                val_x = np.hstack([val_x, self.deg_layers[np.array(val_indices), :]])

            eval_set = [(val_x, val_y)]
        else:
            val_x = None
            val_y = None
            eval_set = []

        if model_name == 'rf':
            if self.split_by == 'material':
                model = RandomForestRegressor(n_jobs=-1, max_depth=6)
            else:
                model = RandomForestRegressor(n_jobs=-1, max_depth=15)
            model.fit(train_x, train_y[:, 0] if train_y.shape[1] == 1 else train_y)
            plot_truth_pred_sklearn(train_x, train_y, val_x, val_y, test_x, test_y, model, self.loss_fn, ax)
        elif model_name == 'svm':
            model = svm.SVR()
            model.fit(train_x, train_y[:, 0] if train_y.shape[1] == 0 else train_y)
            plot_truth_pred_sklearn(train_x, train_y, val_x, val_y, test_x, test_y, model, self.loss_fn, ax)
        elif model_name == 'tabnet':
            from pytorch_tabnet.tab_model import TabNetRegressor
            model = TabNetRegressor(verbose=100)
            model.fit(train_x, train_y, eval_set=eval_set, max_epochs=2000, patience=500, loss_fn=self.loss_fn,
                      eval_metric=[self.loss])
            plot_truth_pred_sklearn(train_x, train_y, val_x, val_y, test_x, test_y, model, self.loss_fn, ax)
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
            prediction, ground_truth, loss = test_tensor(data,
                                                         self._get_additional_tensors_slice(self.test_dataset.indices),
                                                         self.tensors[-1][self.test_dataset.indices, :], self.model,
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
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(111)
        df_all = pd.concat([self.feature_data, self.label_data], axis=1)
        corr = df_all.corr()
        sns.heatmap(corr, ax=ax, annot=True, xticklabels=corr.columns, yticklabels=corr.columns, square=True,
                    cmap='Blues', cbar=False)
        plt.tight_layout()
        plt.savefig(f'../output/{self.project}/corr.pdf')
        if is_notebook():
            plt.show()
        plt.close()

    def _get_additional_tensors_slice(self, indices):
        res = []
        for tensor in self.tensors[1:len(self.tensors) - 1]:
            if tensor is not None:
                res.append(tensor[indices, :])
        return tuple(res)

    def _model_train(self,
                     model,
                     verbose=True,
                     verbose_per_epoch=100,
                     **params,
                     ):
        train_loader = Data.DataLoader(
            self.train_dataset,
            batch_size=int(params["batch_size"]),
            generator=torch.Generator().manual_seed(0)
        )
        val_loader = Data.DataLoader(
            self.val_dataset,
            batch_size=len(self.val_dataset),
            generator=torch.Generator().manual_seed(0)
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
        )

        train_ls = []
        val_ls = []
        stop_epoch = params["epoch"]

        early_stopping = EarlyStopping(
            patience=params["patience"], verbose=False, path=self.ckp_path
        )

        for epoch in range(params["epoch"]):
            train_loss = train(model, train_loader, optimizer, self.loss_fn)
            train_ls.append(train_loss)
            _, _, val_loss = test(model, val_loader, self.loss_fn)
            val_ls.append(val_loss)

            if verbose and ((epoch + 1) % verbose_per_epoch == 0 or epoch == 0):
                print(
                    f"Epoch: {epoch + 1}/{stop_epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Min val loss: {np.min(val_ls):.4f}"
                )

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                if verbose:
                    idx = val_ls.index(min(val_ls))
                    print(
                        f"Early stopping at epoch {epoch + 1}, Checkpoint at epoch {idx + 1}, Train loss: {train_ls[idx]:.4f}, Val loss: {val_ls[idx]:.4f}"
                    )
                break

        idx = val_ls.index(min(val_ls))
        min_loss = val_ls[idx]

        return min_loss, train_ls, val_ls

    def _disable_tqdm(self):
        from functools import partialmethod
        from tqdm import tqdm
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        from tqdm.notebook import tqdm
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        from tqdm.autonotebook import tqdm
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        from tqdm.auto import tqdm
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    def _enable_tqdm(self):
        from functools import partialmethod
        from tqdm import tqdm
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)
        from tqdm.notebook import tqdm
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)
        from tqdm.autonotebook import tqdm
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)
        from tqdm.auto import tqdm
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)
        # reload(sys.modules.get('tqdm.tqdm'))
        # reload(sys.modules.get('tqdm.notebook.tqdm'))
        # reload(sys.modules.get('tqdm.autonotebook.tqdm'))
        # reload(sys.modules.get('tqdm.auto.tqdm'))

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
