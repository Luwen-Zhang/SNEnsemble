import torch
from src.trainer import Trainer
from src.model import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

trainer = Trainer(device=device)
trainer.load_config()
trainer.load_data()
trainer.describe()

models = [
    PytorchTabular(trainer),
    WideDeep(trainer),
    AutoGluon(trainer),
]

trainer.add_modelbases(models)

trainer.get_leaderboard(test_data_only=False, cross_validation=10, verbose=True)
