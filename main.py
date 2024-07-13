import torch
from src.trainer import FatigueTrainer
from src.model import *
import tabensemb
from tabensemb.model import *
from tabensemb.utils import Logging
import os
import argparse

tabensemb._stream_filters = ["DeprecationWarning", "Using batch_size="]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--limit_batch_size",
    type=int,
    required=False,
    default=None,
)
parser.add_argument("--nowrap", dest="nowrap", action="store_true")
parser.add_argument("--use_raw", dest="use_raw", action="store_true")
parser.set_defaults(nowrap=False)
parser.set_defaults(use_raw=False)

args = parser.parse_known_args()[0]
limit_batch_size = args.limit_batch_size
nowrap = args.nowrap
use_raw = args.use_raw

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
        trainer,
        pca=False,
        clustering="KMeans",
        clustering_layer="3L",
        uncertainty="mcd",
        wrap=not nowrap,
        classifier_use_raw=use_raw,
    ),
]
if limit_batch_size is not None:
    for model in models:
        model.limit_batch_size = limit_batch_size
trainer.add_modelbases(models)
trainer.get_leaderboard(cross_validation=10, split_type="random")
log.exit()
