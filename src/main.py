import argparse
from trainer import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))

configfile = 'base_SNL_MSU_DOE_avg'

trainer = Trainer(device=device)
trainer.load_config()
trainer.load_data()

# trainer.params = trainer.bayes()
#
trainer.train()
#
trainer.plot_loss()
trainer.plot_truth_pred()
# trainer.plot_truth_pred_sklearn(model_name='rf')
trainer.plot_feature_importance()
trainer.plot_partial_dependence()
trainer.plot_partial_err()