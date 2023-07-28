import torch
from src.trainer import FatigueTrainer
from src.model import *
from tabensemb.model import *
from tabensemb.utils import Logging
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--reduce_bayes_steps", dest="reduce_bayes_steps", action="store_true"
)
parser.set_defaults(**{"reduce_bayes_steps": False})
args = parser.parse_known_args()[0]
reduce_bayes_steps = args.reduce_bayes_steps

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
    ThisWork(trainer, reduce_bayes_steps=reduce_bayes_steps),
]
trainer.add_modelbases(models)
trainer.get_leaderboard(cross_validation=10, split_type="random")
log.exit()
