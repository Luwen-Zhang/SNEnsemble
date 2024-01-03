import torch
from src.trainer import FatigueTrainer
from src.model import *
import tabensemb
from tabensemb.model import *
from tabensemb.utils import Logging
import os
import argparse

tabensemb._stream_filters = ["DeprecationWarning"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--reduce_bayes_steps", dest="reduce_bayes_steps", action="store_true"
)
parser.add_argument(
    "--investigate_clustering", dest="investigate_clustering", action="store_true"
)
parser.set_defaults(**{"reduce_bayes_steps": False, "investigate_clustering": False})
args = parser.parse_known_args()[0]
reduce_bayes_steps = args.reduce_bayes_steps
investigate_clustering = args.investigate_clustering

log = Logging()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

trainer = FatigueTrainer(device=device)
trainer.load_config()
log.enter(os.path.join(trainer.project_root, "log.txt"))
trainer.summarize_setting()
trainer.load_data()
models = [
    PytorchTabular(trainer),
    WideDeep(trainer),
    AutoGluon(trainer),
    ThisWork(
        trainer, reduce_bayes_steps=reduce_bayes_steps, pca=False, clustering="KMeans"
    ),
]
if investigate_clustering:
    models += [
        ThisWork(
            trainer,
            reduce_bayes_steps=reduce_bayes_steps,
            pca=True,
            clustering="KMeans",
            program="ThisWork_PCA",
        ),
        ThisWork(
            trainer,
            reduce_bayes_steps=reduce_bayes_steps,
            pca=False,
            clustering="GMM",
            program="ThisWork_GMM",
        ),
        ThisWork(
            trainer,
            reduce_bayes_steps=reduce_bayes_steps,
            pca=False,
            clustering="BMM",
            program="ThisWork_BMM",
        ),
    ]
trainer.add_modelbases(models)
trainer.get_leaderboard(cross_validation=10, split_type="random")
log.exit()
