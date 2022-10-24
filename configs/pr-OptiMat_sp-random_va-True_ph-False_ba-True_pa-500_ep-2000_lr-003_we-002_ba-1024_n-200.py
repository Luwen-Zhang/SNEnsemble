class config:
    def __init__(self):
        pass

    data = {
        'project': 'OptiMat',
        'split_by': 'random',
        'validation': True,
        'physics_informed': False,
        'bayes_opt': True,
        'patience': 500,
        'epoch': 2000,
        'lr': 0.003,
        'weight_decay': 0.002,
        'batch_size': 1024,
        'static_params': ['patience', 'epoch'],
        'chosen_params': ['lr', 'weight_decay', 'batch_size'],
        'layers': [16, 64, 128, 128, 64, 16],
        'n_calls': 200,
        'SPACEs': {
            'lr': {'type': 'Real', 'low': 1e-3, 'high': 0.05, 'prior': 'log-uniform'},
            'weight_decay': {'type': 'Real', 'low': 1e-5, 'high': 0.05, 'prior': 'log-uniform'},
            'batch_size': {'type': 'Categorical', 'categories': [32, 64, 128, 256, 512, 1024, 2048, 4096]}
        },
        'feature_names_type': {
            'Width': 1,
            'Area': 1,
            'Percentage of Fibre in 45-deg Direction': 1,
            # 'Percentage of Fibre in 90-deg Direction': 1,
            'Percentage of Fibre in 0-deg Direction': 1,
            'Length(nominal)': 1,
            'Absolute Maximum Stress': 0,
            'Absolute Peak-to-peak Stress': 0,
            'Thickness': 1,
            # 'Frequency': 0,
            # 'Load Length': 1,
            # 'Fibre Volumn Fraction': 1,

        },
        'feature_types': ['Fatigue loading', 'Material'],
        'label_name': ['Cycles to Failure'],

    }


if __name__ == '__main__':
    import shutil

    file_name = ''
    cfg = config()
    for key, value in zip(cfg.data.keys(), cfg.data.values()):
        if not isinstance(value, list) and not isinstance(value, dict):
            short_name = key.split('_')[0][:2]
            short_value = str(value)
            if '.' in short_value:
                short_value = short_value.split('.')[-1]
                if len(short_value) > 4:
                    short_value = short_value[:4]
            file_name += f'_{short_name}-{short_value}'
    file_name = file_name.strip('_')
    shutil.copy(__file__, '../configs/' + file_name + '.py')
    print(file_name)
