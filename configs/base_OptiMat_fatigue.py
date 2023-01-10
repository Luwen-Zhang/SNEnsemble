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
            "database": "OptiMat_fatigue",
            "feature_names_type": {
                "Width": 1,
                "Area": 1,
                "Percentage of Fibre in 45-deg Direction": 1,
                # 'Percentage of Fibre in 90-deg Direction': 1,
                "Percentage of Fibre in 0-deg Direction": 1,
                "Length(nominal)": 1,
                # 'Absolute Maximum Stress': 0,
                # 'Absolute Peak-to-peak Stress': 0,
                "Thickness": 1,
                # 'Frequency': 0,
                # 'Load Length': 1,
                # 'Fibre Volumn Fraction': 1,
            },
            "data_derivers": [
                (
                    "MinStressDeriver",
                    {
                        "derived_name": "Minimum Stress",
                        "stacked": True,
                        "intermediate": True,
                        "max_stress_col": "Maximum Stress",
                        "r_value_col": "R-value",
                    },
                ),
                (
                    "SuppStressDeriver",
                    {
                        "derived_name": "Support Stress",
                        "stacked": True,
                        "max_stress_col": "Maximum Stress",
                        "min_stress_col": "Minimum Stress",
                        "ucs_col": "Static Maximum Compressive Stress",
                        "uts_col": "Static Maximum Tensile Stress",
                        "relative": False,
                    },
                ),
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
