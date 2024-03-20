import torch
from src.trainer import FatigueTrainer
from src.model import *
import tabensemb
from tabensemb.model import *
from tabensemb.utils import Logging
import os
import argparse

tabensemb._stream_filters = ["DeprecationWarning", "PossibleUserWarning"]

log = Logging()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

trainer = FatigueTrainer(device=device)
trainer.load_config()
log.enter(os.path.join(trainer.project_root, "log.txt"))
trainer.summarize_setting()
trainer.load_data()
models = [
    ThisWork(
        trainer,
        clustering="KMeans",
        clustering_layer="1L",
        program="ThisWork_KMeans_1L",
    ),
    ThisWork(
        trainer, clustering="GMM", clustering_layer="1L", program="ThisWork_GMM_1L"
    ),
    ThisWork(
        trainer, clustering="BMM", clustering_layer="1L", program="ThisWork_BMM_1L"
    ),
    ThisWork(
        trainer,
        clustering="KMeans",
        clustering_layer="2L",
        program="ThisWork_KMeans_2L",
    ),
    ThisWork(
        trainer, clustering="GMM", clustering_layer="2L", program="ThisWork_GMM_2L"
    ),
    ThisWork(
        trainer, clustering="BMM", clustering_layer="2L", program="ThisWork_BMM_2L"
    ),
    ThisWork(
        trainer,
        clustering="KMeans",
        clustering_layer="1L",
        program="ThisWork_KMeans_1L_PCA",
        pca=True,
    ),
    ThisWork(
        trainer,
        clustering="GMM",
        clustering_layer="1L",
        program="ThisWork_GMM_1L_PCA",
        pca=True,
    ),
    ThisWork(
        trainer,
        clustering="BMM",
        clustering_layer="1L",
        program="ThisWork_BMM_1L_PCA",
        pca=True,
    ),
    ThisWork(
        trainer,
        clustering="KMeans",
        clustering_layer="2L",
        program="ThisWork_KMeans_2L_PCA",
        pca=True,
    ),
    ThisWork(
        trainer,
        clustering="GMM",
        clustering_layer="2L",
        program="ThisWork_GMM_2L_PCA",
        pca=True,
    ),
    ThisWork(
        trainer,
        clustering="BMM",
        clustering_layer="2L",
        program="ThisWork_BMM_2L_PCA",
        pca=True,
    ),
]
trainer.add_modelbases(models)
trainer.get_leaderboard(cross_validation=10, split_type="random")
log.exit()
