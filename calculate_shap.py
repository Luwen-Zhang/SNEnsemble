import pandas as pd
import torch
import tabensemb
import tabensemb.utils
import os
import pickle
import numpy as np

tabensemb._stream_filters = ["DeprecationWarning", "Deprecated"]

from src.trainer import FatigueTrainer
from tabensemb.trainer import load_trainer, save_trainer

device = "cuda" if torch.cuda.is_available() else "cpu"

trainer = load_trainer("output/analyse/paper_plots/trainer.pkl")

trainer_analysis = load_trainer(
    "output/composite_database_03222024/2024-04-16-23-10-09-0_composite A811 mcd analysis/trainer.pkl"
)

shap_importance_full_train_ls = []
explainers = []
explanations = []

leaderboard = trainer_analysis.leaderboard.sort_values(by="Testing MAPE")
program = "ThisWork"
model_names = leaderboard.loc[leaderboard["Program"] == program, "Model"][:10]
print(model_names)

for model_name in model_names:
    (explainer, explanation, shap_value), feature_names = (
        trainer_analysis.cal_feature_importance(
            program=program,
            model_name=model_name,
            method="shap",
            call_general_method=True,
            indices=trainer_analysis.train_indices,
            return_importance=False,
            explainer="PermutationExplainer",
            call_kwargs=dict(max_evals=10 * len(trainer_analysis.all_feature_names)),
        )
    )
    shap_importance_full_train_ls.append(shap_value)
    explainers.append(explainer)
    explanations.append(explanation)

with open(os.path.join(trainer.project_root, "shap_train_ensemble.pkl"), "wb") as file:
    pickle.dump(
        (explainers, explanations, shap_importance_full_train_ls, feature_names), file
    )


# perm_importance_train_ls = []
# for program, model_name in zip(
#     trainer_analysis.leaderboard["Program"][:10],
#     trainer_analysis.leaderboard["Model"][:10],
# ):
#     perm_importance_train, feature_names = trainer_analysis.cal_feature_importance(
#         program=program,
#         model_name=model_name,
#         method="permutation",
#         call_general_method=True,
#         indices=trainer_analysis.train_indices,
#     )
#     perm_importance_train_ls.append(perm_importance_train)
#
# with open(os.path.join(trainer.project_root, "perm_train.pkl"), "wb") as file:
#     pickle.dump((perm_importance_train_ls, feature_names), file)
