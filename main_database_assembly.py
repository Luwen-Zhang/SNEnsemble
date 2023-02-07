from src.core.trainer import *
from src.core.model import *
from src.core.trainer_assembly import TrainerAssembly

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

configs = [
    "base_SNL_MSU_DOE_fatigue",
    "base_OptiMat_fatigue",
    "base_FACT_fatigue",
    "base_Upwind_fatigue",
]

trainers = []

for config in configs:
    trainer = Trainer(device=device)
    trainer.load_config(config)
    trainer.load_data()

    models = [
        # AutoGluon(trainer),
        # WideDeep(trainer),
        # ModelAssembly(
        #     trainer, models=[TabNet(trainer), MLP(trainer)], program="ThisWorkBaselines"
        # ),
        MLP(trainer),
        # ThisWork(
        #     trainer,
        #     manual_activate=["linlogSN", "loglogSN"],
        #     program="ThisWorklinlog+loglog",
        # ),
    ]

    trainer.add_modelbases(models)

    trainers.append(trainer)

trainer_assem = TrainerAssembly(trainers=trainers)
trainer_assem.eval_all(programs=[model.program for model in models], cross_validation=1)

trainer_assem.plot_loss(model_name="MLP")
