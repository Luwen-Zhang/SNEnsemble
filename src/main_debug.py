from trainer import *

trainer = load_trainer(path='../output/SNL_MSU_DOE_avg_static/base_SNL_MSU_DOE_avg_static/trainer.pkl')
trainer.bayes_opt=True
trainer.pytorch_tabular_tests(verbose=True)