from src.core.trainer import *
from src.core.model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))

configfile = 'base_SNL_MSU_DOE_avg_static'
trainer = Trainer(device=device)
trainer.load_config(configfile)
trainer.load_data()

trainer.describe()

models = [
    # AutoGluon(trainer),
    # PytorchTabular(trainer),
    ThisWork(trainer)
]

trainer.add_modelbases(models)

trainer.train(verbose=True)
trainer.get_leaderboard(test_data_only=False)
