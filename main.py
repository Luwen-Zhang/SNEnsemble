from src.core.trainer import *
from src.core.model import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

trainer = Trainer(device=device)
trainer.load_config()
trainer.load_data()
trainer.describe()

models = [
    AutoGluon(trainer),
    WideDeep(trainer),
    ModelAssembly(
        trainer, models=[TabNet(trainer), MLP(trainer)], program="ThisWorkBaselines"
    ),
    ThisWork(trainer),
]

trainer.add_modelbases(models)

trainer.train(verbose=True)
trainer.get_leaderboard(test_data_only=False)
