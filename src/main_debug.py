from trainer import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))

configfile = 'base_SNL_MSU_DOE_avg_fatigue'
trainer = Trainer(device=device)
trainer.load_config(configfile)
trainer.load_data(selection=True)