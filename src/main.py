import argparse
from trainer import *

parser = argparse.ArgumentParser()
parser.add_argument('--configfile', type=str, required=True)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))

configfile = args.configfile

print(f'\nConfig file: {configfile}\n')

trainer = Trainer(device=device)
trainer.load_config(default_configfile=configfile)
trainer.load_data()

trainer.params = trainer.bayes()

trainer.train()

trainer.plot_loss()
trainer.plot_truth_pred()
trainer.plot_truth_pred_sklearn(model_name='rf')
trainer.plot_feature_importance()
trainer.plot_partial_dependence()
trainer.plot_partial_err()