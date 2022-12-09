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
            "project": "316_creep_fatigue_life_prediction",
            "feature_names_type": {
                "wt.C": 1,
                "wt.Si": 1,
                "wt.Mn": 1,
                "wt.P": 1,
                "wt.S": 1,
                "wt.Ni": 1,
                "wt.Cr": 1,
                "wt.Mo": 1,
                "wt.N": 1,
                "temperature(celsius)": 0,
                "strain amplitude(%)": 0,
                "hold time(h)": 0,
                "strain rate(s-1)": 0,
            },
            "feature_types": ["Fatigue loading", "Material"],
            "label_name": ["log(fatigue life)"],
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
