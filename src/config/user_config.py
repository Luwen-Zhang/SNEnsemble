import sys
from .default_config import DefaultConfig
import json
from typing import Dict


class UserConfig(DefaultConfig):
    def __init__(self, path: str):
        super(UserConfig, self).__init__()
        if path != "base_config":
            json_path = path if "/" in path else f"configs/{path}"
            json_path = (
                json_path if json_path.endswith(".json") else json_path + ".json"
            )
            with open(json_path, "r") as file:
                cfg = json.load(file)
            for key, value in zip(cfg.keys(), cfg.values()):
                if key in self.cfg.keys():
                    self.cfg[key] = value
                else:
                    raise Exception(f'Unexpected item "{key}" in config file.')

    def merge_config(self, config: Dict):
        """
        Merge the input configuration without preventing new keys.

        Parameters
        ----------
        config
            A dict containing configurations.
        """
        for key, value in config.items():
            if value is not None:
                self.cfg[key] = value

    def modify_config(self, config: Dict):
        """
        Modify current configuration using the input config. This operation prevents new keys.

        Parameters
        ----------
        config
            A dict containing configurations.
        """
        for key, value in config.items():
            if key in self.cfg.keys():
                self.cfg[key] = value
            else:
                raise Exception(f"Configuration argument {key} not available.")
