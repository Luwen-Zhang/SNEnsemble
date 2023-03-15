from src.utils import *
from src.model import AbstractModel, TorchModel
from src.trainer import Trainer
from copy import deepcopy as cp


class RFE(TorchModel):
    def __init__(
        self,
        trainer: Trainer,
        modelbase: AbstractModel,
        model_subset=None,
        program=None,
        metric: str = "Validation RMSE",
        impor_method: str = "shap",
        cross_validation=5,
        min_features=1,
        **kwargs,
    ):
        self.metric = metric
        self.impor_method = impor_method
        self.cross_validation = cross_validation
        self.min_features = min_features

        internal_trainer = cp(trainer)
        internal_trainer.modelbases = []
        internal_trainer.modelbases_names = []
        self._model_names = modelbase.get_model_names()
        self.model_class = modelbase.__class__
        super(RFE, self).__init__(
            trainer=trainer, program=program, model_subset=model_subset, **kwargs
        )
        self.trainer_modelbase = {}

        internal_trainer.project_root = self.root

        for model_name in self.get_model_names():
            tmp_trainer = cp(internal_trainer)
            modelbase = self.model_class(tmp_trainer, model_subset=[model_name])
            tmp_trainer.add_modelbases([modelbase])
            self.trainer_modelbase[model_name] = (tmp_trainer, modelbase)
        self.metrics = {}
        self.features_eliminated = {}
        self.selected_features = {}
        self.impor_dicts = {}

    def _get_program_name(self):
        return "RFE-" + self.model_class.__name__

    def _get_model_names(self):
        return self._model_names

    def _new_model(self, model_name, verbose, **kwargs):
        return self.trainer_modelbase[model_name][1].new_model(
            model_name, verbose, **kwargs
        )

    def _predict(self, df: pd.DataFrame, model_name, derived_data=None, **kwargs):
        return self.trainer_modelbase[model_name][1]._predict(
            df, model_name, derived_data, **kwargs
        )

    def _predict_all(self, **kwargs):
        predictions = {}
        for name, (trainer, modelbase) in self.trainer_modelbase.items():
            predictions[name] = modelbase._predict_all(**kwargs)[name]
        return predictions

    def _train(
        self,
        verbose: bool = True,
        model_subset: list = None,
        warm_start=False,
        **kwargs,
    ):
        for model_name in (
            self.get_model_names() if model_subset is None else model_subset
        ):
            if warm_start:
                self.trainer_modelbase[model_name][1]._train(
                    warm_start=warm_start,
                    model_subset=[model_name],
                    verbose=verbose,
                    **kwargs,
                )
            else:
                self.run(verbose=verbose, model_name=model_name)
                self.trainer_modelbase[model_name][1]._train(
                    warm_start=warm_start,
                    model_subset=[model_name],
                    verbose=verbose,
                    **kwargs,
                )

    def run(self, model_name, verbose=True):
        rest_features = cp(self.trainer.all_feature_names)
        trainer, modelbase = self.trainer_modelbase[model_name]
        metrics = []
        features_eliminated = []
        impor_dicts = []
        while len(rest_features) > self.min_features:
            if verbose:
                print(f"Using features: {rest_features}")
            trainer.set_feature_names(rest_features)
            if self.cross_validation == 0:
                modelbase._train(
                    verbose=False, model_subset=[model_name], dump_trainer=False
                )
            leaderboard = trainer.get_leaderboard(
                test_data_only=False,
                cross_validation=self.cross_validation,
                verbose=False,
                dump_trainer=False,
            )
            metrics.append(leaderboard.loc[0, self.metric])
            importance, names = trainer.cal_feature_importance(
                program=modelbase.program,
                model_name=model_name,
                method=self.impor_method,
            )
            impor_dict = {"feature": [], "attr": []}
            for imp, name in zip(importance, names):
                if name in rest_features:
                    impor_dict["feature"].append(name)
                    impor_dict["attr"].append(imp)
            df = pd.DataFrame(impor_dict)
            df.sort_values(by="attr", inplace=True, ascending=False)
            df.reset_index(drop=True, inplace=True)
            rest_features = list(df["feature"])
            print(rest_features)
            features_eliminated.append(rest_features.pop(-1))
            impor_dicts.append(df)
            if verbose:
                print(f"Eliminated feature: {features_eliminated[-1]}")
                # print(f"Permutation importance:\n{df}")

        select_idx = metrics.index(np.min(metrics))
        selected_features = features_eliminated[select_idx:]
        trainer.set_feature_names(selected_features)
        self.metrics[model_name] = metrics
        self.features_eliminated[model_name] = features_eliminated
        self.impor_dicts[model_name] = impor_dicts
        self.selected_features[model_name] = selected_features
        if verbose:
            print(f"Selected features: {selected_features}")
            print(f"Eliminated features: {features_eliminated[:select_idx]}")


class ModelAssembly(AbstractModel):
    def __init__(self, trainer, models=None, program=None, model_subset=None, **kwargs):
        self.program = "ModelAssembly" if program is None else program
        super(ModelAssembly, self).__init__(
            trainer=trainer, program=self.program, model_subset=model_subset, **kwargs
        )
        self.models = {}
        if models is None:
            if model_subset is None:
                raise Exception(f"One of models and model_subset should be specified.")
            else:
                for model_name in model_subset:
                    self.models[model_name] = getattr(
                        sys.modules["src.model"], model_name
                    )(trainer=trainer, model_subset=[model_name], **kwargs)
        else:
            for model in models:
                if len(model.get_model_names()) > 1:
                    raise Exception(
                        f"ModelAssembly is designed for modelbases with a single model."
                    )
                self.models[model.get_model_names()[0]] = model

    def _get_program_name(self):
        return self.program

    def fit(self, model_subset=None, **kwargs):
        for model_name in self.models.keys() if model_subset is None else model_subset:
            self.models[model_name].fit(**kwargs)

    def predict(
        self, df: pd.DataFrame, model_name, derived_data: dict = None, **kwargs
    ):
        return self.models[model_name].predict(
            df=df, model_name=model_name, derived_data=derived_data, **kwargs
        )

    def train(
        self,
        *args,
        **kwargs,
    ):
        print(f"\n-------------Run {self.program}-------------\n")
        self._train(*args, **kwargs)
        print(f"\n-------------{self.program} End-------------\n")

    def _train(self, model_subset=None, *args, **kwargs):
        for model_name in self.models.keys() if model_subset is None else model_subset:
            self.models[model_name]._train(*args, **kwargs)

    def _predict(self, df: pd.DataFrame, model_name, derived_data=None, **kwargs):
        return self.models[model_name].predict(
            df=df, model_name=model_name, derived_data=derived_data, **kwargs
        )

    def _predict_all(self, **kwargs):
        self._check_train_status()
        predictions = {}
        for submodel in self.models.values():
            sub_predictions = submodel._predict_all(**kwargs)
            for key, value in sub_predictions.items():
                predictions[key] = value
        return predictions

    def _get_model_names(self):
        return list(self.models.keys())

    def _check_train_status(self):
        for submodel in self.models.values():
            try:
                submodel._check_train_status()
            except:
                raise Exception(
                    f"{self.program} not trained, run {self.__class__.__name__}.train() first."
                )
