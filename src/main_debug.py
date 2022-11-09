from trainer import *

trainer = load_trainer(path='../output/SNL_MSU_DOE_static/base_SNL_MSU_DOE_static/trainer.pkl')
trainer.bayes_opt=True
trainer.tabnet_tests()