from src.core.trainer import *
from src.core.model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))

# configfile = 'base_SNL_MSU_DOE_avg_fatigue'
# trainer = Trainer(device=device)
# trainer.load_config(configfile)
# trainer.load_data()
#
# trainer.describe()
#
# models = [
#     # AutoGluon(trainer),
#     # PytorchTabular(trainer),
#     # TabNet(trainer),
#     ThisWork(trainer)
# ]
#
# trainer.add_modelbases(models)
#
# trainer.train(verbose=True)
# trainer.get_leaderboard(test_data_only=False)

trainer = load_trainer(path='output/SNL_MSU_DOE_avg_fatigue/2022-12-02-19-05-49_base_SNL_MSU_DOE_avg_fatigue/trainer.pkl')
trainer.plot_partial_dependence(modelbase=trainer._get_modelbase('ThisWork'),log_trans=True, lower_lim=2, upper_lim=7, n_bootstrap=2)
trainer.plot_S_N(s_col='Maximum Stress', n_col='log(Cycles to Failure)', r_col='Minimum/Maximum Stress', m_code='MD-DD5P-UP2[0/±45/0]S', n_bootstrap=1, r_value=10, load_dir='compression')
