import argparse
from tabensemb.trainer import load_trainer
import os
from tabensemb.utils import Logging

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("--cross_validation", default=10, type=int)
args = parser.parse_args()

path: str = args.path
cross_validation = args.cross_validation

if not path.endswith("trainer.pkl"):
    path = os.path.join(path, "trainer.pkl")
if not os.path.isfile(path):
    raise Exception(f"{path} does not exist.")

trainer = load_trainer(path)
log = Logging()
log.enter(os.path.join(trainer.project_root, "log.txt"))
trainer.get_leaderboard(cross_validation=cross_validation, load_from_previous=True)
log.exit()
