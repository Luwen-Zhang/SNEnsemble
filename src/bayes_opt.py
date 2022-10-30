import argparse
from trainer import *

parser = argparse.ArgumentParser()
parser.add_argument('--configfile', type=str, required=True)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))

configfile = args.configfile

trainer = Trainer(device=device)
## Set params
trainer.load_config(default_configfile=configfile)
## Set datasets
trainer.load_data()

trainer.params = trainer.bayes()

print(trainer.params)