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
                    ["IQRRemover", {}],
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

    def test_data_splitter(self):
        import numpy as np
        import pandas as pd
        from src.data.datasplitter import RandomSplitter, MaterialSplitter

        df = pd.DataFrame({"Material_Code": np.random.randint(0, 20, (100,))})
        print("\n-- k-fold RandomSplitter --\n")
        spl = RandomSplitter()
        res_random = [spl.split(df, [], [], [], cv=5) for i in range(5)]
        assert np.allclose(
            np.sort(np.hstack([i[2] for i in res_random])), np.arange(100)
        ), "RandomSplitter is not getting correct k-fold results."

        print("\n-- k-fold RandomSplitter in a new iteration --\n")
        res_random = [spl.split(df, [], [], [], cv=5) for i in range(5)]
        assert np.allclose(
            np.sort(np.hstack([i[2] for i in res_random])), np.arange(100)
        ), "RandomSplitter is not getting correct k-fold results in a new iteration."

        print("\n-- k-fold RandomSplitter change k --\n")
        spl.split(df, [], [], [], cv=5)
        res_random = [spl.split(df, [], [], [], cv=3) for i in range(3)]
        assert np.allclose(
            np.sort(np.hstack([i[2] for i in res_random])), np.arange(100)
        ), "RandomSplitter is not getting correct k-fold results after changing the number of k-fold."

        print("\n-- k-fold MaterialSplitter --\n")
        spl = MaterialSplitter()
        res_mat = [spl.split(df, [], [], [], cv=5) for i in range(5)]
        assert np.allclose(
            np.sort(np.hstack([i[2] for i in res_mat])), np.arange(100)
        ), "MaterialSplitter is not getting correct k-fold results after ."

        print("\n-- Non-cv RandomSplitter --\n")
        spl = RandomSplitter()
        res_random = [spl.split(df, [], [], []) for i in range(5)]
        res = np.sort(np.hstack([i[2] for i in res_random]))
        assert len(res) != 100 or (
            len(res) == 100 and not np.allclose(res, np.arange(100))
        ), "RandomSplitter is getting k-fold results without cv arguments."

        print("\n-- Non-cv MaterialSplitter --\n")
        spl = MaterialSplitter()
        res_mat = [spl.split(df, [], [], []) for i in range(5)]
        res = np.sort(np.hstack([i[2] for i in res_mat]))
        assert len(res) != 100 or (
            len(res) == 100 and not np.allclose(res, np.arange(100))
        ), "MaterialSplitter is getting k-fold results without cv arguments."

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
            CatEmbed(
                trainer,
                model_subset=[
                    "Category Embedding",
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
            program="CatEmbed", model_name="Category Embedding"
        )
        model_trainer.train()
        direct_pred = trainer.get_modelbase("CatEmbed")._predict(
            trainer.datamodule.X_test,
            derived_data=trainer.datamodule.D_test,
            model_name="Category Embedding",
        )
        detached_pred = model_trainer.get_modelbase(
            "CatEmbed_Category Embedding"
        )._predict(
            model_trainer.datamodule.X_test,
            derived_data=model_trainer.datamodule.D_test,
            model_name="Category Embedding",
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
        model_trainer.get_leaderboard(cross_validation=2)

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

    def test_continue_previous(self):
        configfile = "composite_test"
        src.setting["debug_mode"] = True
        trainer = Trainer(device="cpu")
        trainer.load_config(
            configfile,
            manual_config={
                "data_splitter": "RandomSplitter",
            },
        )
        trainer.load_data()

        models = [
            PytorchTabular(trainer, model_subset=["Category Embedding"]),
        ]
        trainer.add_modelbases(models)
        l0 = trainer.get_leaderboard(cross_validation=1, split_type="random")
        l1 = trainer.get_leaderboard(cross_validation=2, load_from_previous=True)

        l2 = trainer.get_leaderboard(cross_validation=2, split_type="random")
        cols = ["Training RMSE", "Testing RMSE", "Validation RMSE"]
        assert np.allclose(
            l1[cols].values.astype(float), l2[cols].values.astype(float)
        ), f"load_from_previous does not get consistent results."

        shutil.rmtree(os.path.join(src.setting["default_output_path"]))

    def test_inspect(self):
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

        print(f"\n-- Initialize model --\n")
        model = CatEmbed(
            trainer,
            model_subset=[
                "Category Embedding",
            ],
        )
        trainer.add_modelbases([model])

        print(f"\n-- Train model --\n")
        trainer.train()

        print(f"\n-- Inspect model --\n")
        direct_inspect = model.inspect_attr(
            "Category Embedding", ["hidden_representation"]
        )
        train_inspect = model.inspect_attr(
            "Category Embedding",
            ["hidden_representation"],
            trainer.df.loc[trainer.train_indices, :],
        )
        train_inspect_with_derived = model.inspect_attr(
            "Category Embedding",
            ["hidden_representation"],
            df=trainer.df.loc[trainer.train_indices, :],
            derived_data=trainer.datamodule.get_derived_data_slice(
                trainer.derived_data, trainer.train_indices
            ),
        )

        assert np.allclose(
            direct_inspect["train"]["prediction"],
            train_inspect["USER_INPUT"]["prediction"],
        )
        assert np.allclose(
            direct_inspect["train"]["prediction"],
            train_inspect_with_derived["USER_INPUT"]["prediction"],
        )
        assert np.allclose(
            direct_inspect["train"]["hidden_representation"],
            train_inspect["USER_INPUT"]["hidden_representation"],
        )
        assert np.allclose(
            direct_inspect["train"]["hidden_representation"],
            train_inspect_with_derived["USER_INPUT"]["hidden_representation"],
        )
        assert not direct_inspect["train"]["hidden_representation"].shape[
            0
        ] == direct_inspect["val"]["hidden_representation"].shape[0] or np.allclose(
            direct_inspect["train"]["hidden_representation"],
            direct_inspect["val"]["hidden_representation"],
        )

    def test_trainer_multitarget(self):
        print(f"\n-- Loading trainer --\n")
        configfile = "composite_test"
        src.setting["debug_mode"] = True
        trainer = Trainer(device="cpu")
        trainer.load_config(
            configfile,
            manual_config={
                "data_splitter": "CycleSplitter",
                "label_name": ["log(Cycles to Failure)", "Cycles to Failure"],
            },
        )
        trainer.load_data()
        trainer.summarize_setting()

        print(f"\n-- Initialize models --\n")

        with self.assertRaises(Exception):
            WideDeep(trainer, model_subset=["TabMlp"])

        models = [
            # PytorchTabular(trainer, model_subset=["Category Embedding"]),
            # AutoGluon(trainer, model_subset=["Linear Regression"]),
            CatEmbed(
                trainer,
                model_subset=[
                    "Category Embedding",
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
            program="CatEmbed", model_name="Category Embedding"
        )
        model_trainer.train()
        direct_pred = trainer.get_modelbase("CatEmbed")._predict(
            trainer.datamodule.X_test,
            derived_data=trainer.datamodule.D_test,
            model_name="Category Embedding",
        )
        detached_pred = model_trainer.get_modelbase(
            "CatEmbed_Category Embedding"
        )._predict(
            model_trainer.datamodule.X_test,
            derived_data=model_trainer.datamodule.D_test,
            model_name="Category Embedding",
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
