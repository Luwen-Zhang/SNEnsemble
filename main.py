import torch
from src.trainer import FatigueTrainer
from src.model import *
from tabensemb.model import *
from tabensemb.utils import Logging
import os

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
    Transformer(trainer),
]
trainer.add_modelbases(models)
trainer.get_leaderboard(cross_validation=10, split_type="random")
log.exit()
