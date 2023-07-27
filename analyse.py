import argparse
from tabensemb.trainer import load_trainer
import os

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
args = parser.parse_args()

path: str = args.path

if not path.endswith("trainer.pkl"):
    path = os.path.join(path, "trainer.pkl")
if not os.path.isfile(path):
    raise Exception(f"{path} does not exist.")

trainer = load_trainer(path)

improved_measure, ttest_res = trainer.get_modelbase("ThisWork").improvement(
    trainer.leaderboard, cv_path=os.path.join(trainer.project_root, "cv")
)
improved_measure.to_csv(os.path.join(trainer.project_root, "improvement.csv"))
method_ranking, detailed = trainer.get_modelbase("ThisWork").method_ranking(
    improved_measure, trainer.leaderboard
)
# method_ranking.to_csv(os.path.join(trainer.project_root, "method_ranking.csv"))
# trainer.get_modelbase("ThisWork").plot_method_ranking(
#     detailed, save_to=os.path.join(trainer.project_root, "method_ranking.jpg")
# )
trainer.get_modelbase("ThisWork").plot_improvement(
    trainer.leaderboard,
    improved_measure,
    ttest_res,
    metric="Testing RMSE",
    save_to=os.path.join(trainer.project_root, "improvement.jpg"),
)

# improved_measure, ttest_res = trainer.get_modelbase("ThisWork").improvement(
#     trainer.leaderboard,
#     cv_path=os.path.join(trainer.project_root, "cv"),
#     exclude=["2L", "GMM", "BMM"],
# )
# improved_measure.to_csv(os.path.join(trainer.project_root, "improvement_exclude.csv"))
# try:
#     method_ranking, detailed = trainer.get_modelbase("ThisWork").method_ranking(
#         improved_measure, trainer.leaderboard, exclude=["2L", "GMM", "BMM"]
#     )
#     method_ranking.to_csv(
#         os.path.join(trainer.project_root, "method_ranking_exclude.csv")
#     )
# except Exception as e:
#     print(e)
# trainer.get_modelbase("ThisWork").plot_method_ranking(
#     detailed, save_to=os.path.join(trainer.project_root, "method_ranking_exclude.jpg")
# )
# trainer.get_modelbase("ThisWork").plot_improvement(
#     trainer.leaderboard,
#     improved_measure,
#     ttest_res,
#     metric="Testing RMSE",
#     save_to=os.path.join(trainer.project_root, "improvement_exclude.jpg"),
# )
