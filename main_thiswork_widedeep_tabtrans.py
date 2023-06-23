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
log.enter(os.path.join(trainer.project_root, "log.txt"))
trainer.summarize_setting()
trainer.load_data()
models = [
    WideDeep(trainer, model_subset=["TabTransformer"]),
    Transformer(
        trainer,
        model_subset=[
            "SNTabTransLR2LPCAKMeans",
            "SNTabTransLR2LPCAGMM",
            "SNTabTransLR2LPCABMM",
        ],
    ),
]
trainer.add_modelbases(models)
trainer.get_leaderboard(cross_validation=10, split_type="random")
log.exit()
