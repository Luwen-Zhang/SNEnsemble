from src.core.trainer_assembly import TrainerAssembly

trainer_paths = [
    'output/SNL_MSU_DOE_avg_fatigue/base_SNL_MSU_DOE_avg_fatigue/trainer.pkl',
    'output/OptiMat_avg_fatigue/base_OptiMat_avg_fatigue/trainer.pkl',
    # 'output/Upwind_avg_fatigue/base_Upwind_avg_fatigue/trainer.pkl',
    # 'output/FACT_avg_fatigue/base_FACT_avg_fatigue/trainer.pkl'
]

ta = TrainerAssembly(trainer_paths=trainer_paths, projects=['SNL-MSU-DOE', 'OptiMat'])

ta.plot_loss(metric='MSE')
ta.eval_all()

# trainer_paths = [
#     'output/SNL_MSU_DOE_avg_static/base_SNL_MSU_DOE_avg_static/trainer.pkl',
#     'output/OptiMat_avg_static/base_OptiMat_avg_static/trainer.pkl',
#     # 'output/Upwind_avg_static/base_Upwind_avg_static/trainer.pkl',
#     # 'output/FACT_avg_static/base_FACT_avg_static/trainer.pkl'
# ]
#
# ta = TrainerAssembly(trainer_paths=trainer_paths, projects=['SNL-MSU-DOE', 'OptiMat'])
#
# ta.plot_loss(metric='MSE')
# ta.eval_all(log_trans=False)