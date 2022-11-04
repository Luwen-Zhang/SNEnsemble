"""
The basic class for the project. It includes configuration, data processing, training, plotting,
and comparing with baseline models.
"""
import os.path
from utils import *
import torch
from torch import nn
import matplotlib.pyplot as plt
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


class Trainer:
    def __init__(self, device='cpu'):
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
            for key, value in parse_res.items():
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

        from copy import deepcopy as cp
        tmp_static_params = cp(self.static_params)
        tmp_static_params['epoch'] = tmp_static_params['epoch'] // 5
        tmp_static_params['patience'] = tmp_static_params['patience'] // 2

        @skopt.utils.use_named_args(self.SPACE)
        def _trainer_bayes_objective(**params):
            res, _, _ = self._model_train(model=self.new_model(),
                                          verbose=False,
                                          **{**params, **tmp_static_params})

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

            # if len(result.func_vals) % 5 == 0:
            #     plt.figure()
            #     ax = plt.subplot(111)
            #     ax = plot_convergence(result, ax)
            #     plt.savefig(f'../output/{self.project}/skopt_convergence.pdf')
            #     plt.close()

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

        print(f'Minimum loss: {min_loss:.5f}')

        test_loader = Data.DataLoader(
            self.test_dataset,
            batch_size=len(self.test_dataset),
            generator=torch.Generator().manual_seed(0),
        )

        _, _, mse = test(self.model, test_loader, torch.nn.MSELoss())
        rmse = np.sqrt(mse)
        self.metrics = {'mse': mse, 'rmse': rmse}

        print(f'Test MSE loss: {mse:.5f}, RMSE loss: {rmse:.5f}')
        save_trainer(self)

    def new_model(self):
        if self.model_name == 'MLP':
            return NN(len(self.feature_names), len(self.label_name), self.layers, self.use_sequence).to(self.device)
        else:
            raise Exception(f'Model {self.model_name} not implemented.')

    '''
    To add new baseline model-bases, implement (i) xxxx_tests for fitting predictors and (ii) _predict_all_pytorch_tabular
    for metrics and plotting, and (iii) branches in get_leaderboard and plot_truth_pred.
    '''

    def autogluon_tests(self, verbose=False, debug_mode=False):
        print('\n-------------Run AutoGluon Tests-------------\n')
        # https://github.com/awslabs/autogluon
        disable_tqdm()
        import warnings
        warnings.simplefilter(action='ignore', category=UserWarning)
        from autogluon.tabular import TabularPredictor
        tabular_dataset = pd.concat([self.feature_data, self.label_data], axis=1)
        predictor = TabularPredictor(label=self.label_name[0], path=self.project_root + 'autogluon')
        with HiddenPrints(disable_logging=True if not verbose else False):
            predictor.fit(tabular_dataset.loc[self.train_dataset.indices, :],
                          tuning_data = tabular_dataset.loc[self.val_dataset.indices, :],
                          presets='best_quality' if not debug_mode else 'medium_quality_faster_train',
                          hyperparameter_tune_kwargs='bayesopt' if not debug_mode else None,
                          use_bag_holdout = True,
                          verbosity=0 if not verbose else 2)
        self.autogluon_leaderboard = predictor.leaderboard(tabular_dataset.loc[self.test_dataset.indices, :],
                                                           silent=True)
        self.autogluon_leaderboard.to_csv(self.project_root + 'autogluon/leaderboard.csv')
        self.autogluon_predictor = predictor
        enable_tqdm()
        warnings.simplefilter(action='default', category=UserWarning)
        save_trainer(self)
        print('\n-------------AutoGluon Tests End-------------\n')

    def pytorch_tabular_tests(self, verbose=False, debug_mode=False):
        print('\n-------------Run Pytorch-tabular Tests-------------\n')
        # https://github.com/manujosephv/pytorch_tabular
        disable_tqdm()
        import warnings
        warnings.simplefilter(action='ignore', category=UserWarning)
        from pytorch_tabular import TabularModel
        from pytorch_tabular.models import CategoryEmbeddingModelConfig, \
            NodeConfig, TabNetModelConfig, TabTransformerConfig, AutoIntConfig, FTTransformerConfig
        from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

        tabular_dataset = pd.concat([self.feature_data, self.label_data], axis=1)

        data_config = DataConfig(
            target=self.label_name,
            continuous_cols=self.feature_names,
            num_workers=8
        )

        trainer_config = TrainerConfig(
            max_epochs=self.static_params['epoch'] if not debug_mode else 10,
            early_stopping_patience=self.static_params['patience'],
        )
        optimizer_config = OptimizerConfig(
        )

        model_configs = [
            CategoryEmbeddingModelConfig(task='regression'),
            NodeConfig(task='regression'),
            TabNetModelConfig(task='regression'),
            TabTransformerConfig(task='regression'),
            AutoIntConfig(task='regression'),
            # FTTransformerConfig(task='regression')
        ]

        # dtype=int because pytorch-tabular can not convert np.int64 to Integer
        # All commented lines in SPACEs cause error. Do not uncomment them.
        SPACEs = {
            'CategoryEmbeddingModel': {'SPACE': [
                Real(low=0, high=0.8, prior='uniform', name='dropout'),  # 0.5
                Real(low=0, high=0.8, prior='uniform', name='embedding_dropout'),  # 0.5
                Real(low=1e-5, high=0.1, prior='log-uniform', name='learning_rate')  # 0.001
            ], 'defaults': [0.5, 0.5, 0.001], 'class': CategoryEmbeddingModelConfig},
            'NODEModel': {'SPACE': [
                Integer(low=2, high=6, prior='uniform', name='depth', dtype=int),  # 6
                Real(low=0, high=0.8, prior='uniform', name='embedding_dropout'),  # 0.0
                Real(low=0, high=0.8, prior='uniform', name='input_dropout'),  # 0.0
                Real(low=1e-5, high=0.1, prior='log-uniform', name='learning_rate'),  # 0.001
                Integer(low=128, high=512, prior='uniform', name='num_trees', dtype=int)
            ], 'defaults': [6, 0.0, 0.0, 0.001, 256], 'class': NodeConfig},
            'TabNetModel': {'SPACE': [
                Integer(low=4, high=64, prior='uniform', name='n_d', dtype=int),  # 8
                Integer(low=4, high=64, prior='uniform', name='n_a', dtype=int),  # 8
                Integer(low=3, high=10, prior='uniform', name='n_steps', dtype=int),  # 3
                Real(low=1.0, high=2.0, prior='uniform', name='gamma'),  # 1.3
                Integer(low=1, high=5, prior='uniform', name='n_independent', dtype=int),  # 2
                Integer(low=1, high=5, prior='uniform', name='n_shared', dtype=int),  # 2
                Real(low=1e-5, high=0.1, prior='log-uniform', name='learning_rate'),  # 0.001
            ], 'defaults': [8, 8, 3, 1.3, 2, 2, 0.001], 'class': TabNetModelConfig},
            'TabTransformerModel': {'SPACE': [
                Real(low=0, high=0.8, prior='uniform', name='embedding_dropout'),  # 0.1
                Real(low=0, high=0.8, prior='uniform', name='ff_dropout'),  # 0.1
                # Categorical([16, 32, 64, 128], name='input_embed_dim'),  # 32
                Real(low=0, high=0.5, prior='uniform', name='shared_embedding_fraction'),  # 0.25
                # Categorical([2, 4, 8, 16], name='num_heads'),  # 8
                Integer(low=4, high=8, prior='uniform', name='num_attn_blocks', dtype=int),  # 6
                Real(low=0, high=0.8, prior='uniform', name='attn_dropout'),  # 0.1
                Real(low=0, high=0.8, prior='uniform', name='add_norm_dropout'),  # 0.1
                Integer(low=2, high=6, prior='uniform', name='ff_hidden_multiplier', dtype=int),  # 4
                Real(low=0, high=0.8, prior='uniform', name='out_ff_dropout'),  # 0.0
                Real(low=1e-5, high=0.1, prior='log-uniform', name='learning_rate'),  # 0.001
            ], 'defaults': [0.1, 0.1, 0.25, 6, 0.1, 0.1, 4, 0.0, 0.001], 'class': TabTransformerConfig},
            'AutoIntModel': {'SPACE': [
                Real(low=1e-5, high=0.1, prior='log-uniform', name='learning_rate'),  # 0.001
                Real(low=0, high=0.8, prior='uniform', name='attn_dropouts'),  # 0.0
                # Categorical([16, 32, 64, 128], name='attn_embed_dim'),  # 32
                Real(low=0, high=0.8, prior='uniform', name='dropout'),  # 0.0
                # Categorical([8, 16, 32, 64], name='embedding_dim'),  # 16
                Real(low=0, high=0.8, prior='uniform', name='embedding_dropout'),  # 0.0
                Integer(low=1, high=4, prior='uniform', name='num_attn_blocks', dtype=int),  # 3
                # Integer(low=1, high=4, prior='uniform', name='num_heads', dtype=int),  # 2
            ], 'defaults': [0.001, 0.0, 0.0, 0.0, 3], 'class': AutoIntConfig},
            'FTTransformerModel': {'SPACE': [
                Real(low=0, high=0.8, prior='uniform', name='embedding_dropout'),  # 0.1
                Real(low=0, high=0.5, prior='uniform', name='shared_embedding_fraction'),  # 0.25
                Integer(low=4, high=8, prior='uniform', name='num_attn_blocks', dtype=int),  # 6
                Real(low=0, high=0.8, prior='uniform', name='attn_dropout'),  # 0.1
                Real(low=0, high=0.8, prior='uniform', name='add_norm_dropout'),  # 0.1
                Real(low=0, high=0.8, prior='uniform', name='ff_dropout'),  # 0.1
                Integer(low=2, high=6, prior='uniform', name='ff_hidden_multiplier', dtype=int),  # 4
                Real(low=0, high=0.8, prior='uniform', name='out_ff_dropout'),  # 0.0
            ], 'defaults': [0.1, 0.25, 6, 0.1, 0.1, 0.1, 4, 0.0], 'class': FTTransformerConfig}
        }

        metrics = {'model': [], 'mse': []}
        models = {}
        for model_config in model_configs:
            model_name = model_config._model_name
            print('Training', model_name)

            SPACE = SPACEs[model_name]['SPACE']
            defaults = SPACEs[model_name]['defaults']
            config_class = SPACEs[model_name]['class']
            exceptions = []
            global _pytorch_tabular_bayes_objective

            @skopt.utils.use_named_args(SPACE)
            def _pytorch_tabular_bayes_objective(**params):
                if verbose:
                    print(params, end=' ')
                with HiddenPrints(disable_logging=True):
                    tabular_model = TabularModel(
                        data_config=data_config,
                        model_config=config_class(task='regression', **params),
                        optimizer_config=optimizer_config,
                        trainer_config=TrainerConfig(
                            max_epochs=self.static_params['epoch'] // 5 if not debug_mode else 10,
                            early_stopping_patience=self.static_params['patience'] // 2,
                        )
                    )
                    tabular_model.config.checkpoints = None
                    tabular_model.config['progress_bar_refresh_rate'] = 0
                    try:
                        tabular_model.fit(train=tabular_dataset.loc[self.train_dataset.indices, :],
                                          validation=tabular_dataset.loc[self.val_dataset.indices, :],
                                          loss=self.loss_fn)
                    except Exception as e:
                        exceptions.append(e)
                        return 1e3
                    res = tabular_model.evaluate(tabular_dataset.loc[self.val_dataset.indices, :])[0][
                        'test_mean_squared_error']
                if verbose:
                    print(res)
                return res

            if not debugger_is_active():
                # otherwise: AssertionError: can only test a child process
                result = gp_minimize(_pytorch_tabular_bayes_objective, SPACE, x0=defaults,
                                     n_calls=self.n_calls if not debug_mode else 11, random_state=0)
                param_names = [x.name for x in SPACE]
                params = {}
                for key, value in zip(param_names, result.x):
                    params[key] = value
                model_config = config_class(task='regression', **params)

                if len(exceptions) > 0 and verbose:
                    print('Exceptions in bayes optimization:', exceptions)

            with HiddenPrints(disable_logging=True if not verbose else False):
                tabular_model = TabularModel(
                    data_config=data_config,
                    model_config=model_config,
                    optimizer_config=optimizer_config,
                    trainer_config=trainer_config
                )
                tabular_model.config.checkpoints_path = self.project_root + 'pytorch_tabular'
                tabular_model.config['progress_bar_refresh_rate'] = 0

                tabular_model.fit(train=tabular_dataset.loc[self.train_dataset.indices, :],
                                  validation=tabular_dataset.loc[self.val_dataset.indices, :],
                                  loss=self.loss_fn)
                metrics['model'].append(tabular_model.config._model_name)
                metrics['mse'].append(tabular_model.evaluate(tabular_dataset.loc[self.test_dataset.indices, :])[0][
                                          'test_mean_squared_error'])
                models[tabular_model.config._model_name] = tabular_model

        metrics = pd.DataFrame(metrics)
        metrics['rmse'] = metrics['mse'] ** (1 / 2)
        metrics.sort_values('rmse', inplace=True)
        self.pytorch_tabular_leaderboard = metrics
        self.pytorch_tabular_models = models

        if verbose:
            print(self.pytorch_tabular_leaderboard)
        self.pytorch_tabular_leaderboard.to_csv(self.project_root + 'pytorch_tabular/leaderboard.csv')

        enable_tqdm()
        warnings.simplefilter(action='default', category=UserWarning)
        save_trainer(self)
        print('\n-------------Pytorch-tabular Tests End-------------\n')

    def get_leaderboard(self, test_data_only=True):
        dfs = []
        metrics = ['rmse', 'mse', 'mae', 'mape', 'r2']
        if hasattr(self, 'pytorch_tabular_leaderboard'):
            predictions = self._predict_all_pytorch_tabular(verbose=False, test_data_only=test_data_only)
            df = Trainer._metrics(predictions, metrics, test_data_only=test_data_only)
            df['Program'] = 'Pytorch-Tabular'
            dfs.append(df)
        if hasattr(self, 'autogluon_leaderboard'):
            predictions = self._predict_all_autogluon(verbose=False, test_data_only=test_data_only)
            df = Trainer._metrics(predictions, metrics, test_data_only=test_data_only)
            df['Program'] = 'AutoGluon'
            dfs.append(df)
        if hasattr(self, 'metrics'):
            predictions = self._predict_all(test_data_only=test_data_only)
            df = Trainer._metrics(predictions, metrics, test_data_only=test_data_only)
            df['Program'] = 'This work'
            dfs.append(df)
        df_leaderboard = pd.concat(dfs, axis=0, ignore_index=True)
        df_leaderboard.sort_values('Test RMSE' if not test_data_only else 'RMSE', inplace=True)
        df_leaderboard.reset_index(drop=True, inplace=True)
        df_leaderboard = df_leaderboard[['Program'] + list(df_leaderboard.columns)[:-1]]
        df_leaderboard.to_csv(self.project_root + 'leaderboard.csv')
        return df_leaderboard

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

    def plot_truth_pred(self, program=None):
        if program is not None:
            print('Making baseline predictions...')
        if program is None:
            model_names = ['--']
            predictions = self._predict_all()
        elif program == 'autogluon':
            if not hasattr(self, 'autogluon_predictor'):
                raise Exception('Autogluon tests have not been run. Run Trainer.autogluon_tests() first.')
            model_names = self.autogluon_leaderboard['model']
            predictions = self._predict_all_autogluon()
        elif program == 'pytorch_tabular':
            if not hasattr(self, 'pytorch_tabular_models'):
                raise Exception('Pytorch-Tabular tests have not been run. Run Trainer.pytorch_tabular_tests() first.')
            model_names = self.pytorch_tabular_leaderboard['model']
            predictions = self._predict_all_pytorch_tabular()
        else:
            raise Exception(f'Program {program} does not exist.')

        if program is not None:
            print('Plotting...')
        for idx, model_name in enumerate(model_names):
            if program is not None:
                print(model_name, f'{idx + 1}/{len(model_names)}')
            plt.figure()
            plt.rcParams['font.size'] = 14
            ax = plt.subplot(111)

            self._plot_truth_pred(predictions, ax, model_name, 'Train', clr[0])
            if 'Validation' in predictions[model_name].keys():
                self._plot_truth_pred(predictions, ax, model_name, 'Validation', clr[2])
            self._plot_truth_pred(predictions, ax, model_name, 'Test', clr[1])

            set_truth_pred(ax)

            plt.legend(loc='upper left', markerscale=1.5, handlelength=0.2, handleheight=0.9)

            s = model_name.replace('/', '_')

            if program is not None:
                plt.savefig(f'../output/{self.project}/{program}/{s}_truth_pred.pdf')
            else:
                plt.savefig(f'../output/{self.project}/truth_pred.pdf')
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

    @staticmethod
    def _metrics(predictions, metrics, test_data_only):
        df_metrics = pd.DataFrame()
        for model_name, model_predictions in predictions.items():
            df = pd.DataFrame(index=[0])
            df['Model'] = model_name
            for tvt, (y_pred, y_true) in model_predictions.items():
                if test_data_only and tvt != 'Test':
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

    def _plot_truth_pred(self, predictions, ax, model_name, name, color):
        pred_y, y = predictions[model_name][name]
        r2 = Trainer._metric_sklearn(y, pred_y, 'r2')
        loss = self.loss_fn(torch.Tensor(y), torch.Tensor(pred_y))
        print(f"{name} Loss: {loss:.4f}, R2: {r2:.4f}")
        ax.scatter(10 ** y, 10 ** pred_y,
                   s=20,
                   color=color,
                   label=f"{name} dataset ($R^2$={r2:.3f})",
                   linewidth=0.4,
                   edgecolors="k")
        ax.set_xlabel("Ground truth")
        ax.set_ylabel("Prediction")

    def _predict_all(self, test_data_only=False):
        train_loader = Data.DataLoader(
            self.train_dataset,
            batch_size=len(self.train_dataset),
            generator=torch.Generator().manual_seed(0),
        )
        val_loader = Data.DataLoader(
            self.val_dataset,
            batch_size=len(self.val_dataset),
            generator=torch.Generator().manual_seed(0),
        )
        test_loader = Data.DataLoader(
            self.test_dataset,
            batch_size=len(self.test_dataset),
            generator=torch.Generator().manual_seed(0),
        )

        if not test_data_only:
            y_train_pred, y_train, _ = test(self.model, train_loader, self.loss_fn)
            y_val_pred, y_val, _ = test(self.model, val_loader, self.loss_fn)
        else:
            y_train_pred = y_train = None
            y_val_pred = y_val = None
        y_test_pred, y_test, _ = test(self.model, test_loader, self.loss_fn)

        predictions = {}
        predictions['--'] = {'Train': (y_train_pred, y_train),
                             'Validation': (y_val_pred, y_val),
                             'Test': (y_test_pred, y_test)}

        return predictions

    def _predict_all_autogluon(self, verbose=True, test_data_only=False):
        model_names = self.autogluon_leaderboard['model']
        tabular_dataset = pd.concat([self.feature_data, self.label_data], axis=1)
        train_data = tabular_dataset.loc[self.train_dataset.indices, :].copy()
        val_data = tabular_dataset.loc[self.val_dataset.indices, :].copy()
        test_data = tabular_dataset.loc[self.test_dataset.indices, :].copy()

        predictions = {}

        for idx, model_name in enumerate(model_names):
            if verbose:
                print(model_name, f'{idx + 1}/{len(model_names)}')
            if not test_data_only:
                y_train_pred = self.autogluon_predictor.predict(train_data, model=model_name, as_pandas=False)
                y_train = train_data[self.autogluon_predictor.label].values

                y_val_pred = self.autogluon_predictor.predict(val_data, model=model_name, as_pandas=False)
                y_val = val_data[self.autogluon_predictor.label].values
            else:
                y_train_pred = y_train = None
                y_val_pred = y_val = None

            y_test_pred = self.autogluon_predictor.predict(test_data, model=model_name, as_pandas=False)
            y_test = test_data[self.autogluon_predictor.label].values

            predictions[model_name] = {'Train': (y_train_pred, y_train), 'Test': (y_test_pred, y_test),
                                       'Validation': (y_val_pred, y_val)}
        return predictions

    def _predict_all_pytorch_tabular(self, verbose=True, test_data_only=False):
        model_names = self.pytorch_tabular_leaderboard['model']
        tabular_dataset = pd.concat([self.feature_data, self.label_data], axis=1)
        train_data = tabular_dataset.loc[self.train_dataset.indices, :].copy()
        val_data = tabular_dataset.loc[self.val_dataset.indices, :].copy()
        test_data = tabular_dataset.loc[self.test_dataset.indices, :].copy()

        predictions = {}
        disable_tqdm()
        for idx, model_name in enumerate(model_names):
            if verbose:
                print(model_name, f'{idx + 1}/{len(model_names)}')
            model = self.pytorch_tabular_models[model_name]

            target = model.config.target[0]

            if not test_data_only:
                y_train_pred = np.array(model.predict(train_data)[f'{target}_prediction'])
                y_train = train_data[target].values

                y_val_pred = np.array(model.predict(val_data)[f'{target}_prediction'])
                y_val = val_data[target].values
            else:
                y_train_pred = y_train = None
                y_val_pred = y_val = None

            y_test_pred = np.array(model.predict(test_data)[f'{target}_prediction'])
            y_test = test_data[target].values

            predictions[model_name] = {'Train': (y_train_pred, y_train), 'Test': (y_test_pred, y_test),
                                       'Validation': (y_val_pred, y_val)}
        enable_tqdm()
        return predictions

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


def save_trainer(trainer, path=None):
    import pickle
    with open(trainer.project_root + 'trainer.pkl' if path is None else path, 'wb') as outp:
        pickle.dump(trainer, outp, pickle.HIGHEST_PROTOCOL)


def load_trainer(path=None):
    import pickle
    with open(path, 'rb') as inp:
        trainer = pickle.load(inp)
    return trainer


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using {} device".format(device))

    trainer = Trainer(device=device)
    trainer.load_config()
    trainer.load_data()

    trainer.train()
    trainer.plot_loss()

    trainer.plot_truth_pred()
    trainer.plot_feature_importance()
    trainer.plot_partial_dependence()
    trainer.plot_partial_err()

    trainer.autogluon_tests(verbose=True)
    trainer.pytorch_tabular_tests(verbose=True)
    trainer.get_leaderboard(test_data_only=True)
    trainer.plot_truth_pred(program='pytorch_tabular')
    trainer.plot_truth_pred(program='autogluon')
