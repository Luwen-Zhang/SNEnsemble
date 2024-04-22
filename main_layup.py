import numpy as np
from src.trainer import FatigueTrainer
from src.model.thiswork_layup import ThisWorkLayup
from tabensemb.model import PytorchTabular, WideDeep, AutoGluon
import tabensemb

# tabensemb.setting["debug_mode"] = True
tabensemb.setting["bayes_loss_limit"] = 5000
tabensemb._stream_filters = ["DeprecationWarning"]

from tabensemb.utils import Logging
import os
import torch


log = Logging()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

trainer = FatigueTrainer(device=device)
trainer.load_config(
    "modulus",
    manual_config={"bayes_opt": True, "patience": 50},
)
log.enter(os.path.join(trainer.project_root, "log.txt"))
trainer.summarize_setting()
trainer.load_data()
models = [
    PytorchTabular(trainer),
    WideDeep(trainer),
    AutoGluon(trainer),
    ThisWorkLayup(trainer),
]
trainer.add_modelbases(models)
# trainer.train()
# trainer.get_leaderboard()
trainer.get_leaderboard(cross_validation=10, split_type="random")
log.exit()
