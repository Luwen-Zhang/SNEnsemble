"""
This is a config file generator. Running the script directly will copy the script itself to the configs folder with
formatted name. This script itself could also be the input of the main script.
"""
import sys

sys.path.append("../configs/")
from base_config import BaseConfig


class config(BaseConfig):
    def __init__(self, do_super=True):
        if do_super:
            super(config, self).__init__()

        cfg = {
            "project": "Upwind_avg_static",
            "feature_names_type": {
                "Thickness": 1,
                "Width": 1,
                "Area": 1,
                "Length(gauge)": 1,
                # 'R-value': 'Minimum/Maximum Stress',
                # 'load (max)[kN]': 'Maximum Load',
                # 'average_emax': 'Average Strain',
                # 'poissonaverage': 'Average Poisson Ratio',
                # 'σmax[MPa]': 'Maximum Stress',
                # 'G[Gpa]': 'Shear Modulus',
                # 'cycles to failurefatigue': 'Cycles to Failure',
                # 'loading rate[mm/min]': 'Displacement Rate',
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
                # 'Relative Maximum Stress': 'Relative Maximum Stress',
                # 'Relative Peak-to-peak Stress': 'Relative Peak-to-peak Stress',
            },
            "data_derivers": {
                'DegLayerDeriver': {'sequence_column': 'Sequence', 'derived_name': 'deg_layers',
                                    'col_names': ['0-deg layers', '45-deg layers', '90-deg layers',
                                                  'Other-deg layers'],
                                    'stacked': True},
            },
            'feature_types': ['Fatigue loading', 'Material', 'Derived'],
            "label_name": ["log(Static Maximum Tensile Stress)"],
        }

        if do_super:
            for key, value in zip(cfg.keys(), cfg.values()):
                if key in self.data.keys():
                    self.data[key] = value
                else:
                    raise Exception(f'Unexpected item "{key}" in config file.')
        else:
            self.data = cfg


if __name__ == "__main__":
    import shutil

    file_name = ""
    cfg = config(do_super=False)
    for key, value in zip(cfg.data.keys(), cfg.data.values()):
        if not isinstance(value, list) and not isinstance(value, dict):
            short_name = key.split("_")[0][:2]
            short_value = str(value)
            if "." in short_value:
                short_value = short_value.split(".")[-1]
                if len(short_value) > 4:
                    short_value = short_value[:4]
            file_name += f"_{short_name}-{short_value}"
    file_name = file_name.strip("_")
    shutil.copy(__file__, "../configs/" + file_name + ".py")
    print(file_name)
