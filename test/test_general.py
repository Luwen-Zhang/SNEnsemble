import unittest
from import_utils import *
import src
from src.config import UserConfig
from src.data import DataModule, AbstractSplitter
import numpy as np
from src.trainer import Trainer, load_trainer, save_trainer
from src.model import *
import torch
from copy import deepcopy as cp
import shutil


class TestGeneral(unittest.TestCase):
    def test_config(self):
        config = UserConfig()
        config.merge(UserConfig.from_file("composite"))
        config = UserConfig("composite")

    def test_datamodule(self):
        print("\n-- Loading config --\n")
        config = UserConfig("composite")
        config.merge(
            {
                "data_processors": [
                    ["CategoricalOrdinalEncoder", {}],
                    ["NaNFeatureRemover", {}],
                    ["VarianceFeatureSelector", {"thres": 1}],
                    ["LackDataMaterialRemover", {}],
                    ["SampleDataAugmentor", {}],
                    ["StandardScaler", {}],
                ],
            }
        )

        print("\n-- Loading datamodule --\n")
        datamodule = DataModule(config=config)

        print("\n-- Loading data --\n")
        datamodule.load_data()

        print(f"\n-- Check splitting --\n")
        AbstractSplitter._check_split(
            datamodule.train_indices,
            datamodule.val_indices,
            datamodule.test_indices,
        )

        print(f"\n-- Check augmentation --\n")
        aug_desc = datamodule.df.loc[
            datamodule.augmented_indices - len(datamodule.dropped_indices),
            datamodule.all_feature_names + datamodule.label_name,
        ].describe()
        original_desc = datamodule.df.loc[
            datamodule.val_indices[-50:],
            datamodule.all_feature_names + datamodule.label_name,
        ].describe()
        assert np.allclose(
            aug_desc.values.astype(float), original_desc.values.astype(float)
        )

        print(f"\n-- Prepare new data when indices are randpermed --\n")
        df = datamodule.df.copy()
        indices = np.array(df.index)
        np.random.shuffle(indices)
        df.index = indices
        df, derived_data = datamodule.prepare_new_data(df)
        assert np.allclose(
            df[datamodule.all_feature_names + datamodule.label_name].values,
            datamodule.df[datamodule.all_feature_names + datamodule.label_name].values,
        ), "Stacked features from prepare_new_data for the set dataframe does not get consistent results"
        assert len(derived_data) == len(datamodule.derived_data), (
            "The number of unstacked features from "
            "prepare_new_data is not consistent"
        )
        for key, value in datamodule.derived_data.items():
            if key != "augmented":
                assert np.allclose(value, derived_data[key]), (
                    f"Unstacked feature `{key}` from prepare_new_data for the set "
                    "dataframe does not get consistent results"
                )

        print(f"\n-- Set feature names --\n")
        datamodule.set_feature_names(datamodule.cont_feature_names[:10])
        assert (
            len(datamodule.cont_feature_names) == 10
            and len(datamodule.cat_feature_names) == 0
            and len(datamodule.label_name) == 1
        ), "set_feature_names is not functional."

        print(f"\n-- Prepare new data after set feature names --\n")
        df, derived_data = datamodule.prepare_new_data(datamodule.df)
        assert (
            len(datamodule.cont_feature_names) == 10
            and len(datamodule.cat_feature_names) == 0
            and len(datamodule.label_name) == 1
        ), "set_feature_names is not functional when prepare_new_data."
        assert np.allclose(
            df[datamodule.all_feature_names + datamodule.label_name].values,
            datamodule.df[datamodule.all_feature_names + datamodule.label_name].values,
        ), (
            "Stacked features from prepare_new_data after set_feature_names for the set dataframe does not get "
            "consistent results"
        )
        assert len(derived_data) == len(datamodule.derived_data), (
            "The number of unstacked features after set_feature_names from "
            "prepare_new_data is not consistent"
        )
        for key, value in datamodule.derived_data.items():
            if key != "augmented":
                assert np.allclose(value, derived_data[key]), (
                    f"Unstacked feature `{key}` after set_feature_names from prepare_new_data for the set "
                    "dataframe does not get consistent results"
                )

        print(f"\n-- Describe --\n")
        datamodule.describe()
        datamodule.cal_corr()

        print(f"\n-- Get not imputed dataframe --\n")
        datamodule.get_not_imputed_df()

    def test_trainer(self):
        print(f"\n-- Loading trainer --\n")
        configfile = "composite_test"
        src.setting["debug_mode"] = True
        trainer = Trainer(device="cpu")
        trainer.load_config(
            configfile,
            manual_config={
                "data_splitter": "CycleSplitter",
            },
        )
        trainer.load_data()
        trainer.summarize_setting()

        print(f"\n-- Initialize models --\n")
        models = [
            PytorchTabular(trainer, model_subset=["Category Embedding"]),
            WideDeep(trainer, model_subset=["TabMlp"]),
            AutoGluon(trainer, model_subset=["Linear Regression"]),
            Transformer(
                trainer,
                model_subset=[
                    "CategoryEmbedding",
                ],
            ),
        ]
        trainer.add_modelbases(models)

        print(f"\n-- Pickling --\n")
        save_trainer(trainer)

        print(f"\n-- Training without bayes --\n")
        trainer.train()

        print(f"\n-- Leaderboard --\n")
        l = trainer.get_leaderboard()

        print(f"\n-- Prediction consistency --\n")
        x_test = trainer.datamodule.X_test
        d_test = trainer.datamodule.D_test
        for model in models:
            model_name = model.model_subset[0]
            pred = model.predict(x_test, model_name=model_name)
            direct_pred = model._predict(
                x_test, derived_data=d_test, model_name=model_name
            )
            assert np.allclose(
                pred, direct_pred
            ), f"{model.__class__.__name__} does not get consistent inference results."

        print(f"\n-- Detach modelbase --\n")
        model_trainer = trainer.detach_model(
            program="Transformer", model_name="CategoryEmbedding"
        )
        model_trainer.train()
        direct_pred = trainer.get_modelbase("Transformer")._predict(
            trainer.datamodule.X_test,
            derived_data=trainer.datamodule.D_test,
            model_name="CategoryEmbedding",
        )
        detached_pred = model_trainer.get_modelbase(
            "Transformer_CategoryEmbedding"
        )._predict(
            model_trainer.datamodule.X_test,
            derived_data=model_trainer.datamodule.D_test,
            model_name="CategoryEmbedding",
        )
        assert np.allclose(
            detached_pred, direct_pred
        ), f"The detached model does not get consistent results."

        print(f"\n-- pytorch cuda functionality --\n")
        if torch.cuda.is_available():
            model_trainer.set_device("cuda")
            model_trainer.train()
        else:
            print(f"Skipping cuda tests since torch.cuda.is_available() is False.")

        print(
            f"\n-- Training after set_feature_names and without categorical features --\n"
        )
        model_trainer.datamodule.set_feature_names(
            model_trainer.datamodule.cont_feature_names[:10]
        )
        model_trainer.train()

        print(f"\n-- Bayes optimization --\n")
        model_trainer.args["bayes_opt"] = True
        model_trainer.train()

        print(f"\n-- Load local trainer --\n")
        root = trainer.project_root + "_rename_test"
        shutil.copytree(trainer.project_root, root)
        shutil.rmtree(trainer.project_root)
        trainer = load_trainer(os.path.join(root, "trainer.pkl"))
        l2 = trainer.get_leaderboard()
        cols = ["Training RMSE", "Testing RMSE", "Validation RMSE"]
        assert np.allclose(
            l[cols].values.astype(float), l2[cols].values.astype(float)
        ), f"Reloaded local trainer does not get consistent results."

        shutil.rmtree(os.path.join(src.setting["default_output_path"]))


if __name__ == "__main__":
    unittest.main()
