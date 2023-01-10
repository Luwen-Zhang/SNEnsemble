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
            "project": "FACT_avg_fatigue",
            "feature_names_type": {
                # 'Lay-up': 1,
                "Percentage of Fibre in 0-deg Direction": 1,
                "Percentage of Fibre in 45-deg Direction": 1,
                "Percentage of Fibre in 90-deg Direction": 1,
                "Percentage of Fibre in Other Direction": 1,
                "Fibre Volumn Fraction": 1,
                # 'Porosity',  ##### Too much absence
                # 'Barcol Hardness',  ##### What is it?
                "Thickness": 1,
                "Maximum Width": 1,
                # 'Minimum Width',
                "Area": 1,
                "Length": 1,
                # 'Load Length',
                # 'Radius of Waist',
                # 'R-value',
                "Maximum Strain": 0,
                "Maximum Stress": 0,
                # 'Static Maximum Tensile Stress',
                # 'Static Maximum Compressive Stress',
                # 'Static Maximum Tensile Strain',
                # 'Static Maximum Compressive Strain',
                # 'Strain Rate',
                "Frequency": 0,
                "Static Elastic Modulus": 1,
                # 'Static Compressive Modulus',
                # 'Temperature',
                # 'Relative Humidity',
            },
            "data_derivers": [
                (
                    "DegLayerDeriver",
                    {
                        "sequence_column": "Sequence",
                        "derived_name": "deg_layers",
                        "col_names": [
                            "0-deg layers",
                            "45-deg layers",
                            "90-deg layers",
                            "Other-deg layers",
                        ],
                        "stacked": True,
                    },
                ),
            ],
            "feature_types": ["Fatigue loading", "Material", "Derived"],
            "label_name": ["log(Cycles to Failure)"],
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
