import os
import sys
import tabensemb

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

tabensemb.setting["default_data_path"] = "../data"
tabensemb.setting["default_config_path"] = "../configs"
tabensemb.setting["default_output_path"] = "../output/unittest"
