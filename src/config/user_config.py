import sys
from .default_config import DefaultConfig
import json


class UserConfig(DefaultConfig):
    def __init__(self, path: str):
        super(UserConfig, self).__init__()
        json_path = path if "/" in path else f"configs/{path}"
        json_path = json_path if json_path.endswith(".json") else json_path + ".json"
        with open(json_path, "r") as file:
            cfg = json.load(file)
        for key, value in zip(cfg.keys(), cfg.values()):
            if key in self.cfg.keys():
                self.cfg[key] = value
            else:
                raise Exception(f'Unexpected item "{key}" in config file.')
