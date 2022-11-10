from trainer import *
from utils import *
import os

class TrainerAssembly:
    def __init__(self, trainer_paths: list, projects=None):
        self.trainers = [load_trainer(path) for path in trainer_paths]
        self.projects = [trainer.project for trainer in self.trainers] if projects is None else projects
        self.project_root = '../output/assembly/' + '_'.join([trainer.project for trainer in self.trainers]) + '/'
        if not os.path.exists('../output/assembly/'):
            os.mkdir('../output/assembly/')
        if not os.path.exists(self.project_root):
            os.mkdir(self.project_root)

    def plot_loss(self, metric=None):
        plt.figure()
        plt.rcParams['font.size'] = 20
        ax = plt.subplot(111)

        for idx, (project, trainer) in enumerate(zip(self.projects, self.trainers)):
            ax.plot(
                np.arange(len(trainer.train_ls)),
                trainer.train_ls,
                label=project + " training loss",
                linewidth=2,
                color=clr[idx],
            )
            ax.plot(
                np.arange(len(trainer.val_ls)),
                trainer.val_ls,
                label=project + " validation loss",
                linewidth=2,
                color=clr[idx],
            )
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f'{self.trainers[0].loss.upper()} Loss' if metric is None else metric + ' Loss')
        plt.savefig(self.project_root + 'loss_epoch.pdf')
        if is_notebook():
            plt.show()
        plt.close()

    def eval_all(self, project_subset: list = None, programs = None, log_trans: bool = True, upper_lim=9):
        """
        Plot all truth_pred plots and get the leaderboard.
        :param project_subset: Choose a list of projects from trainers.
        :param log_trans: Whether the target is log10-transformed. Default to True.
        :param upper_lim: The upper boundary of the plot. Default to 9.
        :return: None
        """
        markers = ['o', 'v', '^', 's', '<', '>']
        dfs = []
        metrics = ['rmse', 'mse', 'mae', 'mape', 'r2']
        programs = ['This work', 'AutoGluon', 'Pytorch-Tabular', 'TabNet'] if programs is None else programs
        for program in programs:
            unique_model_names = []
            all_model_names = []
            all_predictions = []
            for project in project_subset if project_subset is not None else self.projects:
                print(f'Program: {program}, project: {project}')
                trainer = self.trainers[self.projects.index(project)]
                if program == 'This work':
                    model_names = ['--']
                    predictions = trainer._predict_all()
                elif program == 'AutoGluon':
                    if not hasattr(trainer, 'autogluon_predictor'):
                        print(f'No program {program} in project {project}')
                        continue
                    model_names = list(trainer.autogluon_leaderboard['model'])
                    predictions = trainer._predict_all_autogluon(verbose=False)
                elif program == 'Pytorch-Tabular':
                    if not hasattr(trainer, 'pytorch_tabular_models'):
                        print(f'No program {program} in project {project}')
                        continue
                    model_names = list(trainer.pytorch_tabular_leaderboard['model'])
                    predictions = trainer._predict_all_pytorch_tabular(verbose=False)
                elif program == 'TabNet':
                    if not hasattr(trainer, 'tabnet_model'):
                        print(f'No program {program} in project {project}')
                        continue
                    model_names = ['TabNet']
                    predictions = trainer._predict_all_sklearn(trainer.tabnet_model, 'TabNet', verbose=False)
                else:
                    raise Exception(f'Program {program} does not exist.')

                unique_model_names += model_names
                all_model_names.append(model_names)
                all_predictions.append(predictions)

            unique_model_names = list(set(unique_model_names))

            for model_name in unique_model_names:
                plt.figure()
                plt.rcParams['font.size'] = 14
                ax = plt.subplot(111)
                predictions_model = {model_name: {}}
                y_train_pred = []
                y_train_true = []
                y_val_pred = []
                y_val_true = []
                y_test_pred = []
                y_test_true = []
                for proj_idx, (model_names, predictions) in enumerate(zip(all_model_names, all_predictions)):

                    if model_name in model_names:
                        y_train_pred += list(predictions[model_name]['Train'][0].flatten()) if predictions[model_name]['Train'][0] is not None else []
                        y_train_true += list(predictions[model_name]['Train'][1].flatten()) if predictions[model_name]['Train'][1] is not None else []
                        y_val_pred += list(predictions[model_name]['Validation'][0].flatten()) if predictions[model_name]['Validation'][0] is not None else []
                        y_val_true += list(predictions[model_name]['Validation'][1].flatten()) if predictions[model_name]['Validation'][1] is not None else []
                        y_test_pred += list(predictions[model_name]['Test'][0].flatten()) if predictions[model_name]['Test'][0] is not None else []
                        y_test_true += list(predictions[model_name]['Test'][1].flatten()) if predictions[model_name]['Test'][1] is not None else []

                predictions_model[model_name]['Train'] = (np.array(y_train_pred), np.array(y_train_true)) if len(y_train_pred) > 0 else (None, None)
                predictions_model[model_name]['Validation'] = (np.array(y_val_pred), np.array(y_val_true)) if len(y_val_pred) > 0 else (None, None)
                predictions_model[model_name]['Test'] = (np.array(y_test_pred), np.array(y_test_true)) if len(y_test_pred) > 0 else (None, None)

                df = Trainer._metrics(predictions_model, metrics, test_data_only=False)
                df['Program'] = program
                dfs.append(df)

                self.trainers[0]._plot_truth_pred(predictions_model, ax, model_name, 'Train', clr[0], log_trans=log_trans, verbose=False)
                if 'Validation' in predictions_model[model_name].keys():
                    self.trainers[0]._plot_truth_pred(predictions_model, ax, model_name, 'Validation', clr[2],
                                                          log_trans=log_trans, verbose=False)
                self.trainers[0]._plot_truth_pred(predictions_model, ax, model_name, 'Test', clr[1], log_trans=log_trans, verbose=False)

                plt.legend(loc='upper left', markerscale=1.5, handlelength=0.2, handleheight=0.9)

                set_truth_pred(ax, log_trans, upper_lim=upper_lim)

                # plt.legend(loc='upper left', markerscale=1.5, handlelength=0.2, handleheight=0.9)
                s = model_name.replace('/', '_')
                plt.savefig(self.project_root + f'{program}_{s}_truth_pred.pdf')
                if is_notebook():
                    plt.show()

                plt.close()

        df_leaderboard = pd.concat(dfs, axis=0, ignore_index=True)
        df_leaderboard.sort_values('Test RMSE', inplace=True)
        df_leaderboard.reset_index(drop=True, inplace=True)
        df_leaderboard = df_leaderboard[['Program'] + list(df_leaderboard.columns)[:-1]]
        df_leaderboard.to_csv(self.project_root + 'leaderboard.csv')