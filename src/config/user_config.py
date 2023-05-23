from typing import Dict, Union
import json
import os.path
import importlib.machinery
import types
from src.utils import pretty
from .default import cfg as default_cfg


class UserConfig:
    def __init__(self, path: str = None):
        self.cfg = default_cfg
        self.defaults = self.cfg.copy()
        if path is not None:
            self.merge_config(self.from_file(path))

    def available_keys(self):
        return list(self.cfg.keys())

    def defaults(self):
        return self.defaults.copy()

    def merge_config(self, config: Dict):
        """
        Merge the input configuration into the current one.

        Parameters
        ----------
        config
            A dict containing configurations.
        """
        for key, value in config.items():
            if value is not None:
                self.cfg[key] = value

    @staticmethod
    def from_dict(cfg: Dict):
        tmp_cfg = UserConfig()
        tmp_cfg.merge_config(cfg)
        return tmp_cfg

    @staticmethod
    def from_file(path: str) -> Dict:
        file_path = path if "/" in path or os.path.isfile(path) else f"configs/{path}"
        ty = UserConfig.file_type(file_path)
        if ty is None:
            json_path = file_path + ".json"
            py_path = file_path + ".py"
            is_json = os.path.isfile(json_path)
            is_py = os.path.isfile(py_path)
            if is_json and is_py:
                raise Exception(
                    f"Both {json_path} and {py_path} exist. Specify the full name of the file."
                )
            else:
                file_path = json_path if is_json else py_path
                ty = UserConfig.file_type(file_path)
        else:
            if not os.path.isfile(file_path):
                raise Exception(f"{file_path} does not exist.")

        if ty == "json":
            with open(file_path, "r") as file:
                cfg = json.load(file)
        else:
            loader = importlib.machinery.SourceFileLoader("cfg", file_path)
            mod = types.ModuleType(loader.name)
            loader.exec_module(mod)
            cfg = mod.cfg
        return cfg

    def to_file(self, path: str):
        if path.endswith(".json"):
            with open(os.path.join(path), "w") as f:
                json.dump(self.cfg, f, indent=4)
        else:
            if not path.endswith(".py"):
                path += ".py"
            s = "cfg = " + pretty(self.cfg, htchar=" " * 4, indent=0)
            try:
                import black

                s = black.format_str(s, mode=black.Mode())
            except:
                pass
            with open(path, "w") as f:
                f.write(s)

    @staticmethod
    def file_type(path: str) -> Union[str, None]:
        if path.endswith(".json"):
            return "json"
        elif path.endswith(".py"):
            return "py"
        else:
            return None
