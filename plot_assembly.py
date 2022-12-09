from src.core.trainer_assembly import TrainerAssembly

trainer_paths = [
    "output/SNL_MSU_DOE_avg_fatigue/2022-11-23-20-39-50_base_SNL_MSU_DOE_avg_fatigue/trainer.pkl",
    "output/OptiMat_avg_fatigue/2022-11-23-21-24-39_base_OptiMat_avg_fatigue/trainer.pkl",
    "output/Upwind_avg_fatigue/2022-11-23-21-59-12_base_Upwind_avg_fatigue/trainer.pkl",
    "output/FACT_avg_fatigue/2022-11-23-22-33-35_base_FACT_avg_fatigue/trainer.pkl",
]

ta = TrainerAssembly(
    trainer_paths=trainer_paths, projects=["SNL-MSU-DOE", "OptiMat", "Upwind", "FACT"]
)

ta.plot_loss(metric="MSE")
ta.eval_all()

trainer_paths = [
    "output/SNL_MSU_DOE_avg_static/2022-11-23-20-18-19_base_SNL_MSU_DOE_avg_static/trainer.pkl",
    "output/OptiMat_avg_static/2022-11-23-21-07-01_base_OptiMat_avg_static/trainer.pkl",
    "output/Upwind_avg_static/2022-11-23-21-43-19_base_Upwind_avg_static/trainer.pkl",
    "output/FACT_avg_static/2022-11-23-22-19-00_base_FACT_avg_static/trainer.pkl",
]

ta = TrainerAssembly(
    trainer_paths=trainer_paths, projects=["SNL-MSU-DOE", "OptiMat", "Upwind", "FACT"]
)

ta.plot_loss(metric="MSE")
ta.eval_all(log_trans=False)
