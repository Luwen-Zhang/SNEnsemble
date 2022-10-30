"""
This is a base of all other configuration files. It is not a file for the input of the main script.
"""


class BaseConfig:
    def __init__(self):
        self.data = {
            'project': 'FACT',
            'model': 'MLP',
            'loss': 'mse',
            'split_by': 'random',
            'validation': True,
            'bayes_opt': False,
            'patience': 500,
            'epoch': 1000,
            'lr': 0.003,
            'weight_decay': 0.002,
            'batch_size': 1024,
            'static_params': ['patience', 'epoch'],
            'chosen_params': ['lr', 'weight_decay', 'batch_size'],
            'layers': [16, 64, 128, 128, 64, 16],
            'n_calls': 200,
            'sequence': True,
            'SPACEs': {
                'lr': {'type': 'Real', 'low': 1e-3, 'high': 0.05, 'prior': 'log-uniform'},
                'weight_decay': {'type': 'Real', 'low': 1e-5, 'high': 0.05, 'prior': 'log-uniform'},
                'batch_size': {'type': 'Categorical', 'categories': [32, 64, 128, 256, 512, 1024, 2048, 4096]}
            },
            'feature_names_type': {
            },
            'feature_types': ['Fatigue loading', 'Material'],
            'label_name': ['Cycles to Failure'],
        }
