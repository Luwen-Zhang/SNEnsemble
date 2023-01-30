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

    trainer.train(verbose=True)
    leaderboard = trainer.get_leaderboard(test_data_only=True)

    trainer.plot_S_N(
        m_code=m_code,
        s_col="Maximum Stress",
        n_col="log(Cycles to Failure)",
        r_col="R-value",
        n_bootstrap=10,
        r_value=0.1,
        load_dir="tension",
        verbose=False,
        program="MLP",
        model_name="MLP",
        refit=False,
    )
    trainer.plot_S_N(
        m_code=m_code,
        s_col="Maximum Stress",
        n_col="log(Cycles to Failure)",
        r_col="R-value",
        n_bootstrap=10,
        r_value=0.1,
        load_dir="tension",
        verbose=False,
        program=leaderboard["Program"][0]
        if leaderboard["Program"][0] != "MLP"
        else leaderboard["Program"][1],
        refit=False,
    )

    trainers.append(cp(trainer))

trainer_assem = TrainerAssembly(trainers=trainers)
trainer_assem.eval_all(programs=[model.program for model in models], cross_validation=0)

save_trainer_assem(trainer_assem)
