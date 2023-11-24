from tabensemb.utils import *
from tabensemb.model import TorchModel
from ._thiswork.models_clustering_as_input import *
from ._thiswork.models_with_seq import *
from ._thiswork.models_basic import *
from itertools import product
from scipy import stats
from skopt.space import Integer, Real


class ThisWorkAsInput(TorchModel):
    def __init__(self, *args, reduce_bayes_steps=False, **kwargs):
        super(ThisWorkAsInput, self).__init__(*args, **kwargs)
        self.reduce_bayes_steps = reduce_bayes_steps

    def _get_program_name(self):
        return "ThisWorkAsInput"

    @staticmethod
    def _get_model_names():
        return ["AsInput"]

    def _new_model(self, model_name, verbose, required_models=None, **kwargs):
        fix_kwargs = dict(
            n_inputs=len(self.datamodule.cont_feature_names),
            n_outputs=len(self.datamodule.label_name),
            layers=self.datamodule.args["layers"],
            cat_num_unique=[len(x) for x in self.trainer.cat_feature_mapping.values()],
            datamodule=self.datamodule,
        )

        return Abstract1LClusteringModel(
            phy_class=KMeansPhy,
            **fix_kwargs,
            **kwargs,
        )

    def _space(self, model_name):
        return [
            Integer(low=1, high=64, prior="uniform", name="n_clusters", dtype=int),
            Real(low=0.0, high=0.5, prior="uniform", name="mlp_dropout"),
            Real(low=0.0, high=0.5, prior="uniform", name="embed_dropout"),
        ] + self.trainer.SPACE

    def _initial_values(self, model_name):
        res = {
            "n_clusters": 32,
            "mlp_dropout": 0.0,
            "embed_dropout": 0.1,
        }
        res.update(self.trainer.chosen_params)
        return res

    def _custom_training_params(self, model_name) -> Dict:
        if getattr(self, "reduce_bayes_steps", False):
            return dict(epoch=50, bayes_calls=20, bayes_epoch=5)
        else:
            return super(ThisWorkAsInput, self)._custom_training_params(
                model_name=model_name
            )

    def _prepare_custom_datamodule(self, model_name):
        from tabensemb.data import DataModule

        base = self.trainer.datamodule
        datamodule = DataModule(config=self.trainer.datamodule.args, initialize=False)
        datamodule.set_data_imputer("MeanImputer")
        datamodule.set_data_derivers(
            [
                ("UnscaledDataDeriver", {"derived_name": "Unscaled"}),
                # (
                #     "TrendDeriver",
                #     {"stacked": False, "derived_name": "pdp", "plot": True},
                # ),
            ],
        )
        datamodule.set_data_processors(
            [("CategoricalOrdinalEncoder", {}), ("StandardScaler", {})]
        )
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
