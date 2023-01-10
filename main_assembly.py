from src.core.trainer import *
from src.core.model import *
from src.core.trainer_assembly import TrainerAssembly, save_trainer_assem

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

configfile = "base_SNL_MSU_DOE_fatigue"
trainer = Trainer(device=device, project="test")
trainer.load_config(configfile)
trainer.load_data()
m_codes = trainer.get_material_code(unique=True, partition="all").sort_values(
    by="Count", ascending=False
)
selected_m_codes = m_codes["Material_Code"].values[:20]

trainers = []

for m_code in selected_m_codes:
    trainer = Trainer(device=device, project=m_code.replace("/", "_"))
    trainer.load_config(
        configfile, verbose=False, project_root_subfolder="single_material"
    )
    trainer.set_data_processors(
        config=[
            ("MaterialSelector", {"m_code": m_code}),
            ("FeatureValueSelector", {"feature": "R-value", "value": 0.1}),
            ("NaNFeatureRemover", {}),
            ("MeanImputer", {}),
            ("UnscaledDataRecorder", {}),
            ("StandardScaler", {}),
        ]
    )
    trainer.load_data()

    models = [
        # AutoGluon(trainer),
        MLP(trainer),
        ThisWork(
            trainer,
            manual_activate=["linlogSN", "loglogSN"],
            program="ThisWorklinlog+loglog",
        ),
        ThisWork(trainer, manual_activate=["linlogSN"], program="ThisWorklinlog"),
        ThisWork(trainer, manual_activate=["loglogSN"], program="ThisWorkloglog"),
        # ThisWorkPretrain(trainer),
    ]

    trainer.add_modelbases(models)

    trainers.append(cp(trainer))

trainer_assem = TrainerAssembly(trainers=trainers)
trainer_assem.eval_all(programs=[model.program for model in models], cross_validation=5)

save_trainer_assem(trainer_assem)
