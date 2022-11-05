from trainer import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))

trainer = Trainer(device=device)
trainer.load_config()
trainer.load_data()

trainer.params = trainer.bayes()
print(trainer.params)

trainer.train()

trainer.autogluon_tests(verbose=True)
trainer.pytorch_tabular_tests(verbose=True)
trainer.tabnet_tests(verbose=True)
trainer.get_leaderboard(test_data_only=False)
