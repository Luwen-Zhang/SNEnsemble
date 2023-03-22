import torch
from src.trainer import Trainer
from src.model import *
from src.utils import Logging
import os

log = Logging()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

trainer = Trainer(device=device)
trainer.load_config()
os.path.join(trainer.project_root, "log.txt")
trainer.load_data()
models = [
    PytorchTabular(trainer),
    WideDeep(trainer),
    AutoGluon(trainer),
]
trainer.add_modelbases(models)
trainer.train(verbose=True)
trainer.bayes_opt = False
trainer.get_leaderboard(test_data_only=False, cross_validation=10, verbose=True)
log.exit()
