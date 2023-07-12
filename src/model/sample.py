from src.utils import *
from skopt.space import Integer, Categorical, Real
from src.model import TorchModel
from ._transformer.models_basic import *


class CatEmbed(TorchModel):
    def _get_program_name(self):
        return "CatEmbed"

    def _get_model_names(self):
        return ["Category Embedding"]

    def _new_model(self, model_name, verbose, **kwargs):
        fix_kwargs = dict(
            n_inputs=len(self.datamodule.cont_feature_names),
            n_outputs=len(self.datamodule.label_name),
            layers=self.datamodule.args["layers"],
            cat_num_unique=[len(x) for x in self.trainer.cat_feature_mapping.values()],
            datamodule=self.datamodule,
        )
        return CategoryEmbeddingNN(
            **fix_kwargs,
            embedding_dim=3,
            **kwargs,
        )

    def _space(self, model_name):
        return [
            Real(low=0.0, high=0.5, prior="uniform", name="mlp_dropout"),
            Real(low=0.0, high=0.5, prior="uniform", name="embed_dropout"),
        ] + self.trainer.SPACE

    def _initial_values(self, model_name):
        res = {
            "mlp_dropout": 0.0,
            "embed_dropout": 0.1,
        }
        res.update(self.trainer.chosen_params)
        return res

    def _conditional_validity(self, model_name: str) -> bool:
        return True

    def _prepare_custom_datamodule(self, model_name):
        from src.data import DataModule

        base = self.trainer.datamodule
        datamodule = DataModule(config=self.trainer.datamodule.args, initialize=False)
        datamodule.set_data_imputer("MeanImputer")
        datamodule.set_data_derivers(
            [("UnscaledDataDeriver", {"derived_name": "Unscaled"})]
        )
        datamodule.set_data_processors([("StandardScaler", {})])
        datamodule.set_data(
            base.df,
            cont_feature_names=base.cont_feature_names,
            cat_feature_names=base.cat_feature_names,
            label_name=base.label_name,
            train_indices=base.train_indices,
            val_indices=base.val_indices,
            test_indices=base.test_indices,
            verbose=False,
        )
        tmp_derived_data = base.derived_data.copy()
        tmp_derived_data.update(datamodule.derived_data)
        datamodule.derived_data = tmp_derived_data
        self.datamodule = datamodule
        return datamodule

    def _run_custom_data_module(self, df, derived_data, model_name):
        df, my_derived_data = self.datamodule.prepare_new_data(df, ignore_absence=True)
        derived_data = derived_data.copy()
        derived_data.update(my_derived_data)
        derived_data = self.datamodule.sort_derived_data(derived_data)
        return df, derived_data, self.datamodule
