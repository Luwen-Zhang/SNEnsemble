import os.path
from src.utils import *
from src.model import AbstractModel
from src.data import DataModule
from skopt.space import Integer, Categorical, Real
from typing import Dict
import shutil


class AutoGluon(AbstractModel):
    def _get_program_name(self):
        return "AutoGluon"

    def _new_model(self, model_name, verbose, **kwargs):
        from autogluon.tabular import TabularPredictor
        from ._autogluon.multilabel import MultilabelPredictor

        path = os.path.join(self.root, model_name)
        if len(self.trainer.label_name) > 1:
            predictor = MultilabelPredictor(
                labels=self.trainer.label_name,
                path=os.path.join(self.root, model_name),
                problem_types=["regression"] * len(self.trainer.label_name),
            )
        else:
            predictor = TabularPredictor(
                label=self.trainer.label_name[0],
                path=os.path.join(self.root, model_name),
                problem_type="regression",
            )
        if not os.path.exists(path):
            os.mkdir(path)
        return (model_name, predictor)

    def _train_data_preprocess(self):
        data = self.trainer.datamodule
        all_feature_names = self.trainer.all_feature_names
        X_train = data.categories_inverse_transform(data.X_train[all_feature_names])
        X_val = data.categories_inverse_transform(data.X_val[all_feature_names])
        X_test = data.categories_inverse_transform(data.X_test[all_feature_names])
        return {
            "X_train": X_train,
            "y_train": data.y_train,
            "X_val": X_val,
            "y_val": data.y_val,
            "X_test": X_test,
            "y_test": data.y_test,
        }

    def _data_preprocess(self, df, derived_data, model_name):
        all_feature_names = self.trainer.all_feature_names
        df = self.trainer.datamodule.categories_inverse_transform(
            df[all_feature_names].copy()
        )
        return df

    def _train_single_model(
        self,
        model,
        epoch,
        X_train,
        y_train,
        X_val,
        y_val,
        verbose,
        warm_start,
        in_bayes_opt,
        **kwargs,
    ):
        tc = TqdmController()
        tc.disable_tqdm()
        warnings.simplefilter(action="ignore")

        from autogluon.features.generators import PipelineFeatureGenerator
        from autogluon.features.generators.category import CategoryFeatureGenerator
        from autogluon.features.generators.identity import IdentityFeatureGenerator
        from autogluon.common.features.feature_metadata import FeatureMetadata
        from autogluon.common.features.types import R_INT, R_FLOAT

        cont_feature_names = self.trainer.cont_feature_names
        cat_feature_names = self.trainer.cat_feature_names
        label_name = self.trainer.label_name
        feature_metadata = {}
        for feature in cont_feature_names:
            feature_metadata[feature] = "float"
        for feature in cat_feature_names:
            feature_metadata[feature] = "object"
        feature_generator = PipelineFeatureGenerator(
            generators=[
                [
                    IdentityFeatureGenerator(
                        infer_features_in_args=dict(valid_raw_types=[R_INT, R_FLOAT]),
                        feature_metadata_in=FeatureMetadata(feature_metadata),
                    ),
                    CategoryFeatureGenerator(
                        feature_metadata_in=FeatureMetadata(feature_metadata)
                    ),
                ]
            ]
        )
        train_data = X_train.copy()
        train_data[label_name] = y_train
        val_data = X_val.copy()
        val_data[label_name] = y_val
        with HiddenPrints(disable_std=not verbose, disable_logging=not verbose):
            model[1].fit(
                train_data,
                tuning_data=val_data,
                presets="best_quality" if not in_bayes_opt else "medium_quality",
                hyperparameter_tune_kwargs=None if len(kwargs) > 0 else "auto",
                use_bag_holdout=True,
                verbosity=2 if verbose else 0,
                feature_generator=feature_generator,
                hyperparameters={self._name_mapping[model[0]]: kwargs},
                num_gpus=0 if self.device == "cpu" else "auto",
            )
        if not in_bayes_opt:
            model[1].persist_models(max_memory=None)
            if os.path.exists(os.path.join(self.root, model[0])):
                shutil.rmtree(os.path.join(self.root, model[0]))
        tc.enable_tqdm()
        warnings.simplefilter(action="default")

    def _pred_single_model(self, model, X_test, verbose, **kwargs):
        if len(self.trainer.label_name) > 1:
            return model[1].predict(X_test).values
        else:
            return model[1].predict(X_test).values.reshape(-1, 1)

    def _get_model_names(self):
        return [
            "LightGBM",
            "CatBoost",
            "XGBoost",
            "Random Forest",
            "Extremely Randomized Trees",
            "K-Nearest Neighbors",
            "Linear Regression",
            # "Neural Network with MXNet", Deprecated https://github.com/autogluon/autogluon/pull/1603
            "Neural Network with PyTorch",
            "Neural Network with FastAI",
        ]

    @property
    def _name_mapping(self) -> Dict:
        name_mapping = {
            "LightGBM": "GBM",
            "CatBoost": "CAT",
            "XGBoost": "XGB",
            "Random Forest": "RF",
            "Extremely Randomized Trees": "XT",
            "K-Nearest Neighbors": "KNN",
            "Linear Regression": "LR",
            "Neural Network with MXNet": "NN_MXNET",
            "Neural Network with PyTorch": "NN_TORCH",
            "Neural Network with FastAI": "FASTAI",
        }
        return name_mapping

    def _space(self, model_name):
        """
        Spaces are selected according to the official definitions of AutoGluon.
        See autogluon.tabular.predictor.predictor.py for references of each model.
        """
        space_dict = {
            "LightGBM": [
                # Real(low=5e-3, high=0.2, prior="log-uniform", name="learning_rate"),
            ],
            "CatBoost": [
                # Real(low=5e-3, high=0.2, prior="log-uniform", name="learning_rate"),
            ],
            "XGBoost": [
                # Real(low=5e-3, high=0.2, prior="log-uniform", name="learning_rate"),
            ],
            "Random Forest": [],
            "Extremely Randomized Trees": [],
            "K-Nearest Neighbors": [],
            "Linear Regression": [],
            "Neural Network with MXNet": [
                Real(low=1e-4, high=3e-2, prior="log-uniform", name="learning_rate"),
                Real(low=1e-12, high=0.1, prior="log-uniform", name="weight_decay"),
                Real(low=0.0, high=0.5, prior="uniform", name="dropout_prob"),
                Real(low=0.5, high=1.5, prior="uniform", name="embedding_size_factor"),
                Integer(
                    low=4,
                    high=1000,
                    prior="log-uniform",
                    name="proc.embed_min_categories",
                    dtype=int,
                ),
                Integer(
                    low=10,
                    high=10000,
                    prior="log-uniform",
                    name="proc.max_category_levels",
                    dtype=int,
                ),
                Real(low=0.2, high=1.0, prior="uniform", name="proc.skew_threshold"),
                Categorical(categories=[512, 1024, 2056, 128], name="batch_size"),
            ],
            "Neural Network with PyTorch": [
                Real(low=1e-4, high=3e-2, prior="log-uniform", name="learning_rate"),
                Real(low=1e-12, high=0.1, prior="log-uniform", name="weight_decay"),
                Real(low=0.0, high=0.5, prior="uniform", name="dropout_prob"),
                Real(low=0.5, high=1.5, prior="uniform", name="embedding_size_factor"),
                Integer(
                    low=4,
                    high=1000,
                    prior="log-uniform",
                    name="proc.embed_min_categories",
                    dtype=int,
                ),
                Integer(
                    low=10,
                    high=10000,
                    prior="log-uniform",
                    name="proc.max_category_levels",
                    dtype=int,
                ),
                Real(low=0.2, high=1.0, prior="uniform", name="proc.skew_threshold"),
                Integer(
                    low=2,
                    high=4,
                    prior="uniform",
                    name="num_layers",
                    dtype=int,
                ),
                Categorical(categories=[128, 256, 512], name="hidden_size"),
            ],
            "Neural Network with FastAI": [
                Real(low=0.0, high=0.5, prior="uniform", name="emb_drop"),
                Real(low=0.0, high=0.5, prior="uniform", name="ps"),
                Categorical(
                    categories=[256, 64, 128, 512, 1024, 2048, 4096], name="bs"
                ),
                Real(low=5e-5, high=1e-1, prior="log-uniform", name="lr"),
            ],
        }
        return space_dict[model_name]

    def _initial_values(self, model_name):
        params_dict = {
            "LightGBM": {
                # It is sometimes extremely slow to train GBM multiple times on HPC.
                # "learning_rate": 0.03,
            },
            "CatBoost": {
                # "learning_rate": 0.05,
            },
            "XGBoost": {
                # "learning_rate": 0.1,
            },
            "Random Forest": {},
            "Extremely Randomized Trees": {},
            "K-Nearest Neighbors": {},
            "Linear Regression": {},
            "Neural Network with MXNet": {
                "learning_rate": 3e-4,
                "weight_decay": 1e-6,
                "dropout_prob": 0.1,
                "embedding_size_factor": 1.0,
                "proc.embed_min_categories": 4,
                "proc.max_category_levels": 100,
                "proc.skew_threshold": 0.99,
                "batch_size": 512,
            },
            "Neural Network with PyTorch": {
                "learning_rate": 3e-4,
                "weight_decay": 1e-6,
                "dropout_prob": 0.1,
                "embedding_size_factor": 1.0,
                "proc.embed_min_categories": 4,
                "proc.max_category_levels": 100,
                "proc.skew_threshold": 0.99,
                "num_layers": 4,  # number of layers
                "hidden_size": 128,  # number of hidden units in each layer
            },
            "Neural Network with FastAI": {
                "emb_drop": 0.1,
                "ps": 0.1,
                "bs": 256,
                "lr": 1e-2,
            },
        }
        return params_dict[model_name]
