from src.core.trainer import Trainer
from copy import deepcopy as cp
import pandas as pd
import numpy as np


class RecursiveFeatureElimination:
    def __init__(self, trainer: Trainer, model_name: str, metric: str = "RMSE"):
        self.trainer = cp(trainer)
        self.model = self.trainer.get_modelbase(program=model_name)
        self.metric = metric
        self.trainer.modelbases = []
        self.trainer.modelbases_names = []
        self.trainer.add_modelbases([self.model])
        self.metrics = []
        self.features_eliminated = []
        self.impor_dicts = []

    def run(self, cross_validation=5, verbose=True):
        rest_features = list(
            np.setdiff1d(
                self.trainer.feature_names, self.trainer.derived_stacked_features
            )
        )
        while len(rest_features) > 0:
            if verbose:
                print(f"Using features: {rest_features}")
            self.trainer.set_data(
                df=self.trainer.df,
                feature_names=rest_features,
                label_name=self.trainer.label_name,
            )
            leaderboard = self.trainer.get_leaderboard(
                test_data_only=True,
                cross_validation=cross_validation,
                verbose=False,
            )
            self.metrics.append(leaderboard.loc[0, self.metric])
            importance = self.trainer.cal_feature_importance(modelbase=self.model)
            df = pd.DataFrame(
                {
                    "feature": self.trainer.feature_names,
                    "attr": np.abs(importance) / np.sum(np.abs(importance)),
                }
            )
            df.sort_values(by="attr", inplace=True, ascending=False)
            df.reset_index(drop=True, inplace=True)
            rest_features = [
                x
                for x in df["feature"]
                if x not in self.trainer.derived_stacked_features
            ]
            print(rest_features)
            self.features_eliminated.append(rest_features.pop(-1))
            self.impor_dicts.append(df)

            if verbose:
                print(f"Eliminated feature: {self.features_eliminated[-1]}")
                print(f"Permutation importance:\n{df}")
