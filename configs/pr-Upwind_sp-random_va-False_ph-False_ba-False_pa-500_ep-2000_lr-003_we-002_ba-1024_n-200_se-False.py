"""
This is a config file generator. Running the script directly will copy the script itself to the configs folder with
formatted name. This script itself could also be the input of the main script.
"""
import sys

sys.path.append('../configs/')
from base_config import BaseConfig


class config(BaseConfig):
    def __init__(self, do_super=True):
        if do_super:
            super(config, self).__init__()

        cfg = {
            'project': 'Upwind',
            'split_by': 'random',
            'validation': False,
            'physics_informed': False,
            'bayes_opt': False,
            'patience': 500,
            'epoch': 2000,
            'lr': 0.003,
            'weight_decay': 0.002,
            'batch_size': 1024,
            'static_params': ['patience', 'epoch'],
            'chosen_params': ['lr', 'weight_decay', 'batch_size'],
            'layers': [16, 64, 128, 128, 64, 16],
            'n_calls': 200,
            'sequence': False,
            'SPACEs': {
                'lr': {'type': 'Real', 'low': 1e-3, 'high': 0.05, 'prior': 'log-uniform'},
                'weight_decay': {'type': 'Real', 'low': 1e-5, 'high': 0.05, 'prior': 'log-uniform'},
                'batch_size': {'type': 'Categorical', 'categories': [32, 64, 128, 256, 512, 1024, 2048, 4096]}
            },
            'feature_names_type': {
                'Thickness': 1,
                'Width': 1,
                'Area': 1,
                'Length(gauge)': 1,
                # 'R-value': 'Minimum/Maximum Stress',
                # 'load (max)[kN]': 'Maximum Load',
                # 'average_emax': 'Average Strain',
                # 'poissonaverage': 'Average Poisson Ratio',
                # 'σmax[MPa]': 'Maximum Stress',
                # 'G[Gpa]': 'Shear Modulus',
                # 'cycles to failurefatigue': 'Cycles to Failure',
                # 'loading rate[mm/min]': 'Displacement Rate',
                'Frequency': 0,
                # 'E_avg[GPa]': 'Modulus (Tensile or Compressive)',
                # 'T02 max[ºC]': 'Temperature',
                # 'number of layers': 'Number of Layers',
                # 'Plate-Laminate-Width': 'Laminate Width',
                # 'Plate-Fibre Weight': 'Fibre Weight',
                # 'Plate-Laminate-Thickness Adjustment Mould': 'Thickness Adjustment Mould',
                # 'Plate-Laminate-Length': 'Laminate Length',
                # 'Plate-Prepared Resin Mixture-Resin': 'Resin Mixture Resin Weight',
                # 'Plate-Vacuum-Injection': 'Vacuum Injection',
                # 'Plate-Prepared Resin Mixture-Hardener': 'Resin Mixture Hardener Weight',
                # 'Plate-Fibre Weight Percentage': 'Fibre Weight Fraction',
                # 'Plate-Prepared Resin Mixture-Mix': 'Resin Mixture Weight',
                # 'Plate-Prepared Resin Mixture-Ratio': 'Resin Mixture Hardener Fraction',
                # 'Fibre Volumn Fraction': 'Fibre Volumn Fraction',
                # 'Maximum Tensile Stress': 'Maximum Tensile Stress',
                # 'Maximum Compressive Stress': 'Maximum Compressive Stress',
                # 'Static Maximum Tensile Stress': 'Static Maximum Tensile Stress',
                # 'Static Maximum Compressive Stress': 'Static Maximum Compressive Stress',
                # 'Minimum Stress': 'Minimum Stress',
                # 'Static Elastic Modulus': 'Static Elastic Modulus',
                # 'Static Compressive Modulus': 'Static Compressive Modulus',
                'Absolute Maximum Stress': 0,
                'Absolute Peak-to-peak Stress': 0,
                # 'Relative Maximum Stress': 'Relative Maximum Stress',
                # 'Relative Peak-to-peak Stress': 'Relative Peak-to-peak Stress',
            },
            'feature_types': ['Fatigue loading', 'Material'],
            'label_name': ['Cycles to Failure'],

        }

        if do_super:
            for key, value in zip(cfg.keys(), cfg.values()):
                if key in self.data.keys():
                    self.data[key] = value
                else:
                    raise Exception(f'Unexpected item \"{key}\" in config file.')
        else:
            self.data = cfg


if __name__ == '__main__':
    import shutil

    file_name = ''
    cfg = config(do_super=False)
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
