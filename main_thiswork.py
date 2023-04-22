import torch
from src.trainer import Trainer
from src.model import *
from src.utils import Logging
import os
import faulthandler

faulthandler.enable()
log = Logging()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

trainer = Trainer(device=device)
trainer.load_config()
log.enter(os.path.join(trainer.project_root, "log.txt"))
trainer.summarize_setting()
trainer.load_data()
models = [
    Transformer(
        trainer,
        model_subset=[
            "FTTransformer",
            "TransformerLSTM",
            "TransformerSeq",
            "SNTransformer",
            "SNTransformerSeq",
        ],
    ),
]
trainer.add_modelbases(models)
trainer.get_leaderboard(cross_validation=10)
log.exit()
