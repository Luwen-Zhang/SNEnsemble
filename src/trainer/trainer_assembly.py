from src.trainer import Trainer, load_trainer
from src.utils import *
import time
from copy import deepcopy as cp


class TrainerAssembly:
    def __init__(self, trainer_paths: list = None, projects=None, trainers=None):
        self.trainers = (
            [load_trainer(path) for path in trainer_paths]
            if trainers is None
            else trainers
        )
        self.projects = (
            [trainer.project for trainer in self.trainers]
            if projects is None
            else projects
        )
        t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.project_root = (
            f'output/assembly/{t}-{"".join([x[0] for x in self.projects])}/'
        )
        self.leaderboard = None
        if not os.path.exists("output/assembly/"):
            os.mkdir("output/assembly/")
        if not os.path.exists(self.project_root):
            os.mkdir(self.project_root)
        self.projects_program_predictions = None

    def plot_loss(self, metric=None, model_name="ThisWork"):
        plt.figure()
        plt.rcParams["font.size"] = 15
        ax = plt.subplot(111)

        for idx, (project, trainer) in enumerate(zip(self.projects, self.trainers)):
            modelbase = trainer.get_modelbase(model_name)
            name = project.replace("_", " ")
            ax.plot(
                np.arange(len(modelbase.train_ls)),
                modelbase.train_ls,
                label=name + " training loss",
                linewidth=2,
                color=clr[idx],
            )
            ax.plot(
                np.arange(len(modelbase.val_ls)),
                modelbase.val_ls,
                label=name + " validation loss",
                linewidth=2,
                color=clr[idx],
            )
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel(
            f"{self.trainers[0].loss.upper()} Loss"
            if metric is None
            else metric + " Loss"
        )
        plt.savefig(self.project_root + f"{model_name}_loss_epoch.pdf")
        if is_notebook():
            plt.show()
        plt.close()

    def eval_all(
        self,
        project_subset: list = None,
        programs=None,
        log_trans: bool = True,
        upper_lim=9,
        cross_validation=0,
        plots=False,
        re_eval=False,
    ):
        """
        Plot all truth_pred plots and get the leaderboard.
        :param project_subset: Choose a list of projects from trainers.
        :param log_trans: Whether the target is log10-transformed. Default to True.
        :param upper_lim: The upper boundary of the plot. Default to 9.
        :return: None
        """
        markers = ["o", "v", "^", "s", "<", ">"]
        dfs = []
        metrics = ["rmse", "mse", "mae", "mape", "r2"]
        programs = (
            ["ThisWork", "AutoGluon", "PytorchTabular"]
            if programs is None
            else programs
        )

        selected_projects = (
            project_subset if project_subset is not None else self.projects
        )

        if self.projects_program_predictions is None or re_eval:
            projects_program_predictions = {}
            if cross_validation == 0:
                for project in selected_projects:
                    projects_program_predictions[project] = {}
                    trainer = self.trainers[self.projects.index(project)]
                    for program in programs:
                        modelbase = trainer.get_modelbase(program)
                        predictions = modelbase._predict_all(verbose=True)
                        projects_program_predictions[project][program] = predictions
                    trainer._cal_leaderboard(
                        projects_program_predictions[project], test_data_only=True
                    )
            else:
                for project in selected_projects:
                    projects_program_predictions[project] = {}
                    trainer = self.trainers[self.projects.index(project)]
                    projects_program_predictions[project] = trainer.cross_validation(
                        programs=programs,
                        n_random=cross_validation,
                        verbose=True,
                        test_data_only=False,
                    )
                    trainer._cal_leaderboard(
                        projects_program_predictions[project], test_data_only=True
                    )
            self.projects_program_predictions = projects_program_predictions
            self.selected_programs = cp(programs)
            self.selected_projects = cp(selected_projects)
            self.selected_metrics = cp(metrics)
        else:
            projects_program_predictions = self.projects_program_predictions
            programs = self.selected_programs
            selected_projects = self.selected_projects
            metrics = self.selected_metrics
            print(f"Restore loaded setting:")

        print(f"Programs:\n{pretty(programs)}")
        print(f"Projects:\n{pretty(selected_projects)}")
        print(f"Metrics:\n{pretty(metrics)}")

        for program in programs:
            print(f"Evaluate program: {program}")
            unique_model_names = []
            all_model_names = []
            all_predictions = []
            for project in selected_projects:
                trainer = self.trainers[self.projects.index(project)]
                modelbase = trainer.get_modelbase(program)
                model_names = modelbase.get_model_names()
                predictions = projects_program_predictions[project][program]
                unique_model_names += model_names
                all_model_names.append(model_names)
                all_predictions.append(predictions)

            unique_model_names = list(set(unique_model_names))

            for model_name in unique_model_names:
                predictions_model = {model_name: {}}
                y_train_pred = []
                y_train_true = []
                y_val_pred = []
                y_val_true = []
                y_test_pred = []
                y_test_true = []
                for proj_idx, (model_names, predictions) in enumerate(
                    zip(all_model_names, all_predictions)
                ):

                    if model_name in model_names:
                        y_train_pred += (
                            list(predictions[model_name]["Training"][0].flatten())
                            if predictions[model_name]["Training"][0] is not None
                            else []
                        )
                        y_train_true += (
                            list(predictions[model_name]["Training"][1].flatten())
                            if predictions[model_name]["Training"][1] is not None
                            else []
                        )
                        y_val_pred += (
                            list(predictions[model_name]["Validation"][0].flatten())
                            if predictions[model_name]["Validation"][0] is not None
                            else []
                        )
                        y_val_true += (
                            list(predictions[model_name]["Validation"][1].flatten())
                            if predictions[model_name]["Validation"][1] is not None
                            else []
                        )
                        y_test_pred += (
                            list(predictions[model_name]["Testing"][0].flatten())
                            if predictions[model_name]["Testing"][0] is not None
                            else []
                        )
                        y_test_true += (
                            list(predictions[model_name]["Testing"][1].flatten())
                            if predictions[model_name]["Testing"][1] is not None
                            else []
                        )

                predictions_model[model_name]["Training"] = (
                    (np.array(y_train_pred), np.array(y_train_true))
                    if len(y_train_pred) > 0
                    else (None, None)
                )
                predictions_model[model_name]["Validation"] = (
                    (np.array(y_val_pred), np.array(y_val_true))
                    if len(y_val_pred) > 0
                    else (None, None)
                )
                predictions_model[model_name]["Testing"] = (
                    (np.array(y_test_pred), np.array(y_test_true))
                    if len(y_test_pred) > 0
                    else (None, None)
                )

                df = Trainer._metrics(predictions_model, metrics, test_data_only=False)
                df["Program"] = program
                dfs.append(df)

                if plots:
                    plt.figure()
                    plt.rcParams["font.size"] = 14
                    ax = plt.subplot(111)

                    self.trainers[0]._plot_truth_pred(
                        predictions_model,
                        ax,
                        model_name,
                        "Training",
                        clr[0],
                        log_trans=log_trans,
                        verbose=False,
                    )
                    if "Validation" in predictions_model[model_name].keys():
                        self.trainers[0]._plot_truth_pred(
                            predictions_model,
                            ax,
                            model_name,
                            "Validation",
                            clr[2],
                            log_trans=log_trans,
                            verbose=False,
                        )
                    self.trainers[0]._plot_truth_pred(
                        predictions_model,
                        ax,
                        model_name,
                        "Testing",
                        clr[1],
                        log_trans=log_trans,
                        verbose=False,
                    )

                    plt.legend(
                        loc="upper left",
                        markerscale=1.5,
                        handlelength=0.2,
                        handleheight=0.9,
                    )

                    set_truth_pred(ax, log_trans, upper_lim=upper_lim)

                    # plt.legend(loc='upper left', markerscale=1.5, handlelength=0.2, handleheight=0.9)
                    s = model_name.replace("/", "_")
                    plt.savefig(self.project_root + f"{program}_{s}_truth_pred.pdf")
                    if is_notebook():
                        plt.show()

                    plt.close()

        df_leaderboard = pd.concat(dfs, axis=0, ignore_index=True)
        df_leaderboard.sort_values("Testing RMSE", inplace=True)
        df_leaderboard.reset_index(drop=True, inplace=True)

        model_existence = df_leaderboard[["Program", "Model"]]
        model_existence = pd.concat(
            [model_existence, pd.DataFrame(columns=selected_projects)], axis=1
        )
        all_model_names = [
            program + model
            for program, model in zip(
                df_leaderboard["Program"], df_leaderboard["Model"]
            )
        ]
        for project in selected_projects:
            all_exist_model_names = []
            trainer = self.trainers[self.projects.index(project)]
            for program in programs:
                modelbase = trainer.get_modelbase(program)
                model_names = modelbase.get_model_names()
                all_exist_model_names += [program + name for name in model_names]

            unique_exist_model_names = list(set(all_exist_model_names))
            for model_name in unique_exist_model_names:
                model_existence.loc[all_model_names.index(model_name), project] = 1

        df_leaderboard = pd.concat(
            [df_leaderboard, model_existence[selected_projects]], axis=1
        )
        columns_ahead = ["Program", "Model"] + selected_projects
        columns_back = list(sorted(np.setdiff1d(df_leaderboard.columns, columns_ahead)))

        df_leaderboard = df_leaderboard[columns_ahead + columns_back]
        df_leaderboard.to_csv(self.project_root + "leaderboard.csv")
        self.leaderboard = df_leaderboard


def save_trainer_assem(trainer_assem, path=None, verbose=True):
    import pickle

    path = trainer_assem.project_root + "trainer_assem.pkl" if path is None else path
    if verbose:
        print(
            f"TrainerAssembly saved. To load the trainer_assem, run trainer_assem = load_trainer_assem(path='{path}')"
        )
    with open(path, "wb") as outp:
        pickle.dump(trainer_assem, outp, pickle.HIGHEST_PROTOCOL)


def load_trainer_assem(path=None):
    import pickle

    with open(path, "rb") as inp:
        trainer_assem = pickle.load(inp)
    return trainer_assem
