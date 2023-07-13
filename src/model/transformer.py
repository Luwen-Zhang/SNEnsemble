import warnings
from src.utils import *
from skopt.space import Integer, Categorical, Real
from src.model import TorchModel
from ._transformer.models_clustering import *
from ._transformer.models_with_seq import *
from ._transformer.models_basic import *
from itertools import product


class Transformer(TorchModel):
    def _get_program_name(self):
        return "Transformer"

    @staticmethod
    def _get_model_names():
        available_names = []
        try:
            from .autogluon import AutoGluon

            available_names += [f"AutoGluon_{x}" for x in AutoGluon._get_model_names()]
        except:
            pass
        try:
            from .widedeep import WideDeep

            available_names += [f"WideDeep_{x}" for x in WideDeep._get_model_names()]
        except:
            pass
        try:
            from .pytorch_tabular import PytorchTabular

            available_names += [
                f"PytorchTabular_{x}" for x in PytorchTabular._get_model_names()
            ]
        except:
            pass

        all_names = [
            "_".join(x)
            for x in product(
                available_names,
                ["Wrap", "NoWrap"],
                ["1L", "2L"],
                ["PCA"],
                ["KMeans"],
            )
        ]
        return all_names

    def _new_model(self, model_name, verbose, required_models=None, **kwargs):
        fix_kwargs = dict(
            n_inputs=len(self.datamodule.cont_feature_names),
            n_outputs=len(self.datamodule.label_name),
            layers=self.datamodule.args["layers"],
            cat_num_unique=[len(x) for x in self.trainer.cat_feature_mapping.values()],
            datamodule=self.datamodule,
        )
        components = model_name.split("_")
        if "Wrap" in components:
            cont_cat_model = required_models[
                f"EXTERN_{components[0]}_{components[1]}_WRAP"
            ]
        else:
            cont_cat_model = required_models[f"EXTERN_{components[0]}_{components[1]}"]
        if "1L" in components:
            cls = Abstract1LClusteringModel
            if "KMeans" in components:
                sn_class = KMeansSN
            elif "GMM" in components:
                sn_class = GMMSN
            elif "BMM" in components:
                sn_class = BMMSN
            else:
                raise Exception(f"Clustering algorithm not found.")
        else:
            cls = Abstract2LClusteringModel
            if "KMeans" in components:
                sn_class = TwolayerKMeansSN
            elif "GMM" in components:
                sn_class = TwolayerGMMSN
            elif "BMM" in components:
                sn_class = TwolayerBMMSN
            else:
                raise Exception(f"Clustering algorithm not found.")
        if "PCA" in components:
            if "1L" in components:
                feature_idx = cls.basic_clustering_features_idx(self.datamodule)
            else:
                feature_idx = cls.top_clustering_features_idx(self.datamodule)
            if len(feature_idx) > 2:
                pca = self.datamodule.pca(feature_idx=feature_idx)
                n_pca_dim = (
                    np.where(pca.explained_variance_ratio_.cumsum() < 0.9)[0][-1] + 1
                )
            else:
                n_pca_dim = len(feature_idx)
        else:
            n_pca_dim = None

        return cls(
            sn_class=sn_class,
            **fix_kwargs,
            embedding_dim=3,
            n_pca_dim=n_pca_dim,
            cont_cat_model=cont_cat_model,
            **kwargs,
        )

    def _space(self, model_name):
        components = model_name.split("_")
        if "1L" in components:
            return [
                Integer(low=1, high=64, prior="uniform", name="n_clusters", dtype=int),
            ] + self.trainer.SPACE
        elif "2L" in components:
            return [
                Integer(low=1, high=64, prior="uniform", name="n_clusters", dtype=int),
                Integer(
                    low=1,
                    high=32,
                    prior="uniform",
                    name="n_clusters_per_cluster",
                    dtype=int,
                ),
            ] + self.trainer.SPACE

    def _initial_values(self, model_name):
        components = model_name.split("_")
        res = {}
        if "1L" in components:
            res = {"n_clusters": 16}
        elif "2L" in components:
            res = {"n_clusters": 16, "n_clusters_per_cluster": 8}
        res.update(self.trainer.chosen_params)
        return res

    def _conditional_validity(self, model_name: str) -> bool:
        components = model_name.split("_")
        if "Wrap" in components and any([model in components for model in ["TabNet"]]):
            return False
        if "Wrap" in components and "AutoGluon" in components:
            return False
        if "Wrap" in components and "PytorchTabular_NODE" in model_name:
            return False
        if (
            "2L" in components
            and len(
                AbstractClusteringModel.top_clustering_features_idx(
                    self.trainer.datamodule
                )
            )
            == 0
        ):
            return False
        return True

    def required_models(self, model_name: str) -> Union[List[str], None]:
        components = model_name.split("_")
        if "Wrap" in components:
            models = [f"EXTERN_{components[0]}_{components[1]}_WRAP"]
        else:
            models = [f"EXTERN_{components[0]}_{components[1]}"]
        return models

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

    def inspect_weighted_predictions(self, model_name, **kwargs):
        model = self.model[model_name]
        target_attr = ["dl_weight", "dl_pred", "phy_pred"]
        if not hasattr(model, "dl_weight"):
            raise Exception(
                f"The model does not have the attribute `dl_weight`. Is it trained?"
            )
        if hasattr(model, "uncertain_dl_weight"):
            target_attr += ["uncertain_dl_weight", "mu", "std"]
        inspect_dict = self.inspect_attr(
            model_name=model_name, attributes=target_attr, **kwargs
        )
        inspect_dict = {
            key: val
            if not isinstance(val, torch.Tensor)
            else val.detach().cpu().numpy()
            for key, val in inspect_dict.items()
        }
        return inspect_dict

    def plot_uncertain_dl_weight(self, inspect_dict, save_to=None):
        import matplotlib.ticker as ticker

        def plot_once(dl_weight, mu, std, title, ax):
            sorted_idx = np.argsort(dl_weight.flatten())
            x = np.arange(1, len(dl_weight) + 1)
            ax.scatter(
                x,
                dl_weight[sorted_idx],
                c="#D81159",
                label="Truth",
            )
            ax.errorbar(
                x,
                mu[sorted_idx],
                yerr=std[sorted_idx],
                fmt="co",
                mfc="#0496FF",
                mec="#0496FF",
                ecolor="#0496FF",
                # capsize=5,
                label="GPR prediction (±std)",
            )
            ax.legend(fontsize="x-small", loc="upper left")
            ax.set_title(title)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.set_xlim([0.5, len(dl_weight) + 1.5])
            ax.set_ylim([0, 1])

        def plot_part(part_dict, title, ax):
            dl_weight, mu, std = (
                part_dict["dl_weight"],
                part_dict["mu"],
                part_dict["std"],
            )
            plot_once(dl_weight, mu, std, title, ax)

        fig = plt.figure(figsize=(12, 4))
        ax = plt.subplot(131)
        plot_part(inspect_dict["train"], "Training set", ax)
        ax = plt.subplot(132)
        plot_part(inspect_dict["val"], "Validation set", ax)
        ax = plt.subplot(133)
        plot_part(inspect_dict["test"], "Testing set", ax)
        ax = fig.add_subplot(111, frameon=False)
        plt.tick_params(
            labelcolor="none",
            which="both",
            top=False,
            bottom=False,
            left=False,
            right=False,
        )
        ax.set_ylabel("Deep learning weight")
        ax.set_xlabel("Indices of data points (sorted by the target value)")
        plt.tight_layout()
        if save_to is not None:
            plt.savefig(save_to, dpi=500)
        plt.show()
        plt.close()

    # def _bayes_eval(
    #     self,
    #     model,
    #     X_train,
    #     y_train,
    #     X_val,
    #     y_val,
    # ):
    #     """
    #     Evaluating the model for bayesian optimization iterations. The larger one of training and evaluation errors is
    #     returned.
    #
    #     Returns
    #     -------
    #     result
    #         The evaluation of bayesian hyperparameter optimization.
    #     """
    #     y_val_pred = self._pred_single_model(model, X_val, verbose=False)
    #     val_loss = metric_sklearn(y_val_pred, y_val, self.trainer.args["loss"])
    #     y_train_pred = self._pred_single_model(model, X_train, verbose=False)
    #     train_loss = metric_sklearn(y_train_pred, y_train, self.trainer.args["loss"])
    #     return max([train_loss, val_loss])
