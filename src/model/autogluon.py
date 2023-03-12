import os.path
from src.utils import *
from src.model import AbstractModel
from src.trainer import save_trainer
from skopt.space import Integer, Categorical, Real
from typing import Dict


class AutoGluon(AbstractModel):
    def _get_program_name(self):
        return "AutoGluon"

    def _new_model(self, model_name, verbose, **kwargs):
        from autogluon.tabular import TabularPredictor

        path = os.path.join(self.root, model_name)
        predictor = TabularPredictor(
            label=self.trainer.label_name[0],
            path=os.path.join(self.root, model_name),
            problem_type="regression",
        )
        if not os.path.exists(path):
            os.mkdir(path)
        return (model_name, predictor)

    def _train_data_preprocess(
        self,
        X_train,
        D_train,
        y_train,
        X_val,
        D_val,
        y_val,
        X_test,
        D_test,
        y_test,
    ):
        all_feature_names = self.trainer.all_feature_names
        X_train = X_train[all_feature_names]
        X_val = X_val[all_feature_names]
        X_test = X_test[all_feature_names]
        return X_train, D_train, y_train, X_val, D_val, y_val, X_test, D_test, y_test

    def _data_preprocess(self, df, derived_data, model_name):
        all_feature_names = self.trainer.all_feature_names
        label_name = self.trainer.label_name
        df = df[all_feature_names + label_name].copy()
        return df, derived_data

    def _train_single_model(
        self,
        model,
        epoch,
        X_train,
        D_train,
        y_train,
        X_val,
        D_val,
        y_val,
        verbose,
        warm_start,
        **kwargs,
    ):
        disable_tqdm()
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
        train_data[label_name[0]] = y_train
        val_data = X_val.copy()
        val_data[label_name[0]] = y_val
        with HiddenPrints(disable_logging=True):
            model[1].fit(
                train_data,
                tuning_data=val_data,
                presets="best_quality",
                hyperparameter_tune_kwargs=None,
                use_bag_holdout=True,
                verbosity=0,
                feature_generator=feature_generator,
                hyperparameters={self._name_mapping[model[0]]: kwargs},
            )

        enable_tqdm()
        warnings.simplefilter(action="default")

    def _pred_single_model(self, model, X_test, D_test, verbose, **kwargs):
        return model[1].predict(X_test)

    def _get_model_names(self):
        return [
            "LightGBM",
            "CatBoost",
            "XGBoost",
            "Random Forest",
            "Extremely Randomized Trees",
            "K-Nearest Neighbors",
            "Linear Regression",
            "Neural Network with MXNet",
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
        space_dict = {
            "LightGBM": [
                Real(low=5e-3, high=0.2, prior="log-uniform", name="learning_rate"),
            ],
            "CatBoost": [
                Real(low=5e-3, high=0.2, prior="log-uniform", name="learning_rate"),
            ],
            "XGBoost": [
                Real(low=5e-3, high=0.2, prior="log-uniform", name="learning_rate"),
            ],
            "Random Forest": [],
            "Extremely Randomized Trees": [],
            "K-Nearest Neighbors": [],
            "Linear Regression": [],
            "Neural Network with MXNet": [
                Real(low=1e-4, high=3e-2, prior="log-uniform", name="learning_rate"),
                Real(low=1e-12, high=0.1, prior="log-uniform", name="weight_decay"),
                Categorical(
                    categories=[0.1, 0.0, 0.5, 0.2, 0.3, 0.4], name="dropout_prob"
                ),
                Categorical(
                    categories=[1.0, 0.5, 1.5, 0.7, 0.6, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4],
                    name="embedding_size_factor",
                ),
                Categorical(
                    categories=[4, 3, 10, 100, 1000], name="proc.embed_min_categories"
                ),
                Categorical(
                    categories=[100, 10, 20, 200, 300, 400, 500, 1000, 10000],
                    name="proc.max_category_levels",
                ),
                Categorical(
                    categories=[0.99, 0.2, 0.3, 0.5, 0.8, 0.9, 0.999, 1.0, 10.0, 100.0],
                    name="proc.skew_threshold",
                ),
                Categorical(categories=[512, 1024, 2056, 128], name="batch_size"),
            ],
            "Neural Network with PyTorch": [
                Real(low=1e-4, high=3e-2, prior="log-uniform", name="learning_rate"),
                Real(low=1e-12, high=0.1, prior="log-uniform", name="weight_decay"),
                Categorical(
                    categories=[0.1, 0.0, 0.5, 0.2, 0.3, 0.4], name="dropout_prob"
                ),
                Categorical(
                    categories=[1.0, 0.5, 1.5, 0.7, 0.6, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4],
                    name="embedding_size_factor",
                ),
                Categorical(
                    categories=[4, 3, 10, 100, 1000], name="proc.embed_min_categories"
                ),
                Categorical(
                    categories=[100, 10, 20, 200, 300, 400, 500, 1000, 10000],
                    name="proc.max_category_levels",
                ),
                Categorical(
                    categories=[0.99, 0.2, 0.3, 0.5, 0.8, 0.9, 0.999, 1.0, 10.0, 100.0],
                    name="proc.skew_threshold",
                ),
                Categorical(categories=[2, 3, 4], name="num_layers"),
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
            "LightGBM": {"learning_rate": 0.03},
            "CatBoost": {"learning_rate": 0.05},
            "XGBoost": {"learning_rate": 0.1},
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
