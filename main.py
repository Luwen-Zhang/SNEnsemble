import torch
from src.trainer import Trainer
from src.model import *
from src.utils import Logging
import os
import faulthandler

log = Logging()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

trainer = Trainer(device=device)
trainer.load_config()
log.enter(os.path.join(trainer.project_root, "log.txt"))
faulthandler.enable(open(os.path.join(trainer.project_root, "fault_log.txt"), "a"))
trainer.summarize_setting()
trainer.load_data()
models = [
    PytorchTabular(trainer),
    WideDeep(trainer),
    AutoGluon(trainer),
]
trainer.add_modelbases(models)
trainer.get_leaderboard(cross_validation=10)
faulthandler.disable()
log.exit()
