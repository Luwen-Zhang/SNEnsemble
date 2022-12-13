from src.core.trainer import *
from src.core.model import *
from src.core.trainer_assembly import TrainerAssembly, save_trainer_assem

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

trainers = []
models = []

configfiles = [
    "base_SNL_MSU_DOE_fatigue",
    "base_OptiMat_fatigue",
    "base_Upwind_fatigue",
    "base_FACT_fatigue",
]

for configfile in configfiles:
    trainer = Trainer(device=device)
    trainer.load_config(configfile)

    models = [
        AutoGluon(trainer),
        PytorchTabular(trainer),
        ModelAssembly(
            trainer, models=[TabNet(trainer), MLP(trainer)], program="ThisWorkBaselines"
        ),
        ThisWork(trainer),
    ]

    trainer.add_modelbases(models)

    trainers.append(trainer)

trainer_assem = TrainerAssembly(trainers=trainers)
trainer_assem.plot_loss(metric="MSE")
trainer_assem.eval_all(programs=[model.program for model in models], cross_validation=5)

save_trainer_assem(trainer_assem)
