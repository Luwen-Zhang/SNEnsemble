from trainer import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))

configfile = 'base_SNL_MSU_DOE_avg'

trainer = Trainer(device=device)
trainer.load_config(default_configfile=configfile)
trainer.load_data()

# trainer.params = trainer.bayes()
# print(trainer.params)
#
# trainer.train()
#
# if find_executable('latex'):
#     trainer.plot_loss()
#     trainer.plot_truth_pred()
#     trainer.plot_feature_importance()
#     trainer.plot_partial_dependence()
#     trainer.plot_partial_err()

# trainer.autogluon_tests(verbose=True, debug_mode=True)
trainer.pytorch_tabular_tests(verbose=True, debug_mode=True)
trainer.get_leaderboard(test_data_only=True)

if find_executable('latex'):
    trainer.plot_truth_pred(program='pytorch_tabular')
    trainer.plot_truth_pred(program='autogluon')