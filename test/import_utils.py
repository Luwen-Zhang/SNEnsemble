import os
import sys

current_work_dir = os.path.abspath(".")
parent_dir = os.path.split(current_work_dir)[-1]

if parent_dir == "test":
    sys.path.append("../")
try:
    import src
except:
    raise Exception(
        f"Test units should be placed in a folder named `test` that is in the same parent folder as `src`."
    )

src.setting["default_data_path"] = "../data"
src.setting["default_config_path"] = "../configs"
src.setting["default_output_path"] = "../output/unittest"
