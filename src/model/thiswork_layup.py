from .thiswork import ThisWork
from tabensemb.model import TorchModel
from skopt.space import Integer, Categorical, Real
from itertools import product
from typing import List, Union
from ._thiswork_layup.transformer import AbstractLayupModel, TransformerLayup


class ThisWorkLayup(ThisWork):
    def _get_program_name(self):
        return "ThisWorkLayup"

    @staticmethod
    def _get_model_names():
        available_names = ThisWork._get_other_model_base_model_names()

        all_names = ["TransformerLayup"] + [
            "_".join(x)
            for x in product(
                available_names,
                ["NoWrap", "Wrap"],
            )
        ]
        for name in all_names.copy():
            components = name.split("_")
            wrap_invalid = (
                any([model in components for model in ["TabNet"]])
                or "AutoGluon" in components
                or "PytorchTabular_NODE" in name
            )
            if "PytorchTabular_TabTransformer" in name:
                pass
            if "Wrap" in components and wrap_invalid:
                all_names.remove(name)
            elif "NoWrap" in components and not wrap_invalid:
                all_names.remove(name)
        # all_names += ["CatEmbed_Category Embedding_Wrap_1L_NoPCA_KMeans"]
        return all_names

    def _new_model(self, model_name, verbose, required_models=None, **kwargs):
        datamodule = self.trainer.datamodule
        fix_kwargs = dict(layers=datamodule.args["layers"], datamodule=datamodule)
        if model_name == "TransformerLayup":
            return TransformerLayup(**fix_kwargs, **kwargs)
        else:
            components = model_name.split("_")
            if "Wrap" in components:
                cont_cat_model = required_models[
                    f"EXTERN_{components[0]}_{components[1]}_WRAP"
                ]
            else:
                cont_cat_model = required_models[
                    f"EXTERN_{components[0]}_{components[1]}"
                ]
            seq_model = required_models["TransformerLayup"]
            return AbstractLayupModel(
                **fix_kwargs,
                seq_model=seq_model,
                cont_cat_model=cont_cat_model,
                **kwargs,
            )

    def _initial_values(self, model_name):
        if model_name == "TransformerLayup":
            res = {
                "seq_attn_heads": 4,
                "seq_attn_layers": 2,
                "seq_embedding_dim": 32,
                "attn_ff_dim": 32,
                "seq_attn_dropout": 0.1,
            }
        else:
            res = {"dropout": 0.0}
        res.update(self.trainer.chosen_params)
        return res

    def _conditional_validity(self, model_name: str) -> bool:
        if model_name != "TransformerLayup":
            components = model_name.split("_")
            if (
                components[0] not in self.trainer.modelbases_names
                or components[1]
                not in self.trainer.get_modelbase(
                    program=components[0]
                ).get_model_names()
            ):
                return False
            return True
        else:
            return True

    def _space(self, model_name):
        return (
            [
                Categorical(categories=[2, 4, 8, 16, 32], name="seq_attn_heads"),
                Categorical(categories=[2, 4, 8, 16, 32], name="seq_attn_layers"),
                Categorical(categories=[32, 64, 128], name="seq_embedding_dim"),
                Categorical(categories=[16, 32, 64, 128, 256, 512], name="attn_ff_dim"),
                Real(low=1e-9, high=0.3, prior="uniform", name="seq_attn_dropout"),
            ]
            if model_name == "TransformerLayup"
            else [Real(low=0, high=0.3, prior="uniform", name="dropout")]
        ) + self.trainer.SPACE

    def required_models(self, model_name: str) -> Union[List[str], None]:
        if model_name == "TransformerLayup":
            return None
        else:
            components = model_name.split("_")
            if "Wrap" in components:
                models = [f"EXTERN_{components[0]}_{components[1]}_WRAP"]
            else:
                models = [f"EXTERN_{components[0]}_{components[1]}"]
            models += ["TransformerLayup"]
            return models
