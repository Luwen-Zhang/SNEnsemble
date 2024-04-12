from tabensemb.utils import *
from tabensemb.model import TorchModel
from ._thiswork.models_clustering import *
from itertools import product
from scipy import stats
from skopt.space import Integer, Real
from typing import Union, List, Iterable
from ._thiswork.clustering.singlelayer import GMMPhy, BMMPhy, KMeansPhy
from ._thiswork.clustering.multilayer import (
    TwolayerKMeansPhy,
    TwolayerBMMPhy,
    TwolayerGMMPhy,
    MultilayerKMeansPhy,
    MultilayerBMMPhy,
    MultilayerGMMPhy,
)


class ThisWork(TorchModel):
    def __init__(
        self,
        *args,
        reduce_bayes_steps=False,
        n_pca_dim=None,
        pca=False,
        clustering="KMeans",
        clustering_layer="1L",
        **kwargs,
    ):
        self.reduce_bayes_steps = reduce_bayes_steps
        self.n_pca_dim = n_pca_dim
        self.pca = pca
        self.clustering = clustering
        self.clustering_layer = clustering_layer
        super(ThisWork, self).__init__(*args, **kwargs)

    def _get_program_name(self):
        return "ThisWork"

    @staticmethod
    def _get_other_model_base_model_names():
        available_names = []
        try:
            from tabensemb.model.autogluon import AutoGluon

            available_names += [f"AutoGluon_{x}" for x in AutoGluon._get_model_names()]
        except:
            pass
        try:
            from tabensemb.model.widedeep import WideDeep

            available_names += [f"WideDeep_{x}" for x in WideDeep._get_model_names()]
        except:
            pass
        try:
            from tabensemb.model.pytorch_tabular import PytorchTabular

            available_names += [
                f"PytorchTabular_{x}" for x in PytorchTabular._get_model_names()
            ]
        except:
            pass
        return available_names

    @staticmethod
    def _get_model_names():
        available_names = ThisWork._get_other_model_base_model_names()

        physics_names = [
            "PHYSICS_" + "_".join(x)
            for x in product(
                ["NoPCA", "PCA"],
                ["1L", "2L", "3L"],
                ["KMeans", "GMM", "BMM"],
            )
        ]
        all_names = physics_names + [
            "_".join(x)
            for x in product(
                available_names,
                ["NoWrap", "Wrap"],
                ["NoPCA", "PCA"],
                ["1L", "2L", "3L"],
                ["KMeans", "GMM", "BMM"],
            )
        ]
        for name in all_names.copy():
            components = name.split("_")
            if "PHYSICS" not in components:
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
        components = model_name.split("_")
        if "PHYSICS" not in components:
            cls = AbstractClusteringModel
            if "Wrap" in components:
                cont_cat_model = required_models[
                    f"EXTERN_{components[0]}_{components[1]}_WRAP"
                ]
            else:
                cont_cat_model = required_models[
                    f"EXTERN_{components[0]}_{components[1]}"
                ]
            phy_name = f"PHYSICS_{'_'.join(components[-3:])}"
            clustering_phy_model = required_models[phy_name]
            return cls(
                datamodule=self.datamodule,
                embedding_dim=3,
                cont_cat_model=cont_cat_model,
                clustering_phy_model=clustering_phy_model,
                phy_name=phy_name,
                **kwargs,
            )
        else:
            if "KMeans" in components:
                phy_class = {
                    "1L": KMeansPhy,
                    "2L": TwolayerKMeansPhy,
                    "3L": MultilayerKMeansPhy,
                }[self.clustering_layer]
            elif "GMM" in components:
                phy_class = {
                    "1L": GMMPhy,
                    "2L": TwolayerGMMPhy,
                    "3L": MultilayerGMMPhy,
                }[self.clustering_layer]
            elif "BMM" in components:
                phy_class = {
                    "1L": BMMPhy,
                    "2L": TwolayerBMMPhy,
                    "3L": MultilayerBMMPhy,
                }[self.clustering_layer]
            else:
                raise Exception(f"Clustering algorithm not found.")

            if "PCA" in components:
                if "1L" == self.clustering_layer:
                    feature_idx = phy_class.basic_clustering_features_idx(
                        self.datamodule
                    )
                else:
                    feature_idx = phy_class.first_clustering_features_idx(
                        self.datamodule
                    )
                if len(feature_idx) > 2:
                    pca = self.datamodule.pca(feature_idx=feature_idx)
                    n_pca_dim = (
                        (
                            np.where(pca.explained_variance_ratio_.cumsum() < 0.9)[0][
                                -1
                            ]
                            + 1
                        )
                        if self.n_pca_dim is None
                        else self.n_pca_dim
                    )
                else:
                    n_pca_dim = len(feature_idx)
            else:
                n_pca_dim = None

            if "1L" == self.clustering_layer:
                return phy_class(
                    n_pca_dim=n_pca_dim,
                    datamodule=self.datamodule,
                    on_cpu=True,
                    **kwargs,
                )
            elif "2L" == self.clustering_layer:
                clustering_features = list(
                    phy_class.basic_clustering_features_idx(self.datamodule)
                )
                top_level_clustering_features = phy_class.top_clustering_features_idx(
                    self.datamodule
                )
                input_2_idx = [
                    list(clustering_features).index(x)
                    for x in top_level_clustering_features
                ]
                input_1_idx = list(
                    np.setdiff1d(np.arange(len(clustering_features)), input_2_idx)
                )
                return phy_class(
                    n_pca_dim=n_pca_dim,
                    datamodule=self.datamodule,
                    n_input_1=len(input_1_idx),
                    n_input_2=len(input_2_idx),
                    input_1_idx=input_1_idx,
                    input_2_idx=input_2_idx,
                    on_cpu=True,
                    **kwargs,
                )
            else:
                clustering_features = list(
                    phy_class.basic_clustering_features_idx(self.datamodule)
                )
                input_idx_1 = [
                    clustering_features.index(x)
                    for x in phy_class.first_clustering_features_idx(self.datamodule)
                ]
                input_idx_2 = [
                    clustering_features.index(x)
                    for x in phy_class.second_clustering_features_idx(self.datamodule)
                ]
                input_idx_3 = [
                    clustering_features.index(x)
                    for x in phy_class.third_clustering_features_idx(self.datamodule)
                ]
                return phy_class(
                    datamodule=self.datamodule,
                    n_clusters_ls=[
                        kwargs["n_clusters_1"],
                        kwargs["n_clusters_2"],
                        kwargs["n_clusters_3"],
                    ],
                    input_idxs=[input_idx_1, input_idx_2, input_idx_3],
                    kwargses=[dict(n_pca_dim=n_pca_dim), {}, {}],
                    on_cpu=True,
                    **kwargs,
                )

    def _space(self, model_name):
        if "PHYSICS" in model_name:
            if "1L" == self.clustering_layer:
                return [
                    Integer(
                        low=1, high=256, prior="uniform", name="n_clusters", dtype=int
                    ),
                    Real(low=1e-8, high=1e8, prior="log-uniform", name="l1_penalty"),
                    self.trainer.SPACE[
                        list(self.trainer.chosen_params.keys()).index("batch_size")
                    ],
                ]
            elif "2L" == self.clustering_layer:
                return [
                    Integer(
                        low=1, high=256, prior="uniform", name="n_clusters", dtype=int
                    ),
                    Integer(
                        low=1,
                        high=32,
                        prior="uniform",
                        name="n_clusters_per_cluster",
                        dtype=int,
                    ),
                    Real(low=1e-8, high=1e8, prior="log-uniform", name="l1_penalty"),
                    self.trainer.SPACE[
                        list(self.trainer.chosen_params.keys()).index("batch_size")
                    ],
                ]
            else:
                return [
                    Integer(
                        low=1, high=256, prior="uniform", name="n_clusters_1", dtype=int
                    ),
                    Integer(
                        low=1,
                        high=32,
                        prior="uniform",
                        name="n_clusters_2",
                        dtype=int,
                    ),
                    Integer(
                        low=1,
                        high=16,
                        prior="uniform",
                        name="n_clusters_3",
                        dtype=int,
                    ),
                    Real(low=1e-8, high=1e8, prior="log-uniform", name="l1_penalty"),
                    self.trainer.SPACE[
                        list(self.trainer.chosen_params.keys()).index("batch_size")
                    ],
                ]
        else:
            return [
                Real(low=0, high=0.3, prior="uniform", name="dropout")
            ] + self.trainer.SPACE

    def _initial_values(self, model_name):
        if "PHYSICS" in model_name:
            if "1L" == self.clustering_layer:
                return {
                    "n_clusters": 32,
                    "l1_penalty": 1e-8,
                    "batch_size": self.trainer.chosen_params["batch_size"],
                }
            elif "2L" == self.clustering_layer:
                return {
                    "n_clusters": 32,
                    "n_clusters_per_cluster": 8,
                    "l1_penalty": 1e-8,
                    "batch_size": self.trainer.chosen_params["batch_size"],
                }
            else:
                return {
                    "n_clusters_1": 32,
                    "n_clusters_2": 8,
                    "n_clusters_3": 8,
                    "l1_penalty": 1e-8,
                    "batch_size": self.trainer.chosen_params["batch_size"],
                }
        else:
            res = {"dropout": 0.0}
            res.update(self.trainer.chosen_params)
            return res

    def _custom_training_params(self, model_name) -> Dict:
        if getattr(self, "reduce_bayes_steps", False):
            return dict(epoch=50, bayes_calls=20, bayes_epoch=5)
        else:
            return super(ThisWork, self)._custom_training_params(model_name=model_name)

    def _conditional_validity(self, model_name: str) -> bool:
        if self.pca and "NoPCA" in model_name:
            return False
        if not self.pca and "NoPCA" not in model_name and "PCA" in model_name:
            return False
        if self.clustering not in model_name:
            return False
        if self.clustering_layer not in model_name:
            return False
        if "PHYSICS" not in model_name:
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

    def required_models(self, model_name: str) -> Union[List[str], None]:
        components = model_name.split("_")
        if "PHYSICS" not in model_name:
            if "Wrap" in components:
                models = [f"EXTERN_{components[0]}_{components[1]}_WRAP"]
            else:
                models = [f"EXTERN_{components[0]}_{components[1]}"]
            models += [f"PHYSICS_{'_'.join(components[-3:])}"]
            return models
        else:
            return None

    def _prepare_custom_datamodule(self, model_name, warm_start=False):
        from tabensemb.data import DataModule

        base = self.trainer.datamodule
        if not warm_start or not hasattr(self, "datamodule"):
            datamodule = DataModule(
                config=self.trainer.datamodule.args, initialize=False
            )
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
            datamodule.set_data_processors([("StandardScaler", {})])
            warm_start = False
        else:
            datamodule = self.datamodule
        datamodule.set_data(
            base.df,
            cont_feature_names=base.cont_feature_names,
            cat_feature_names=base.cat_feature_names,
            label_name=base.label_name,
            train_indices=base.train_indices,
            val_indices=base.val_indices,
            test_indices=base.test_indices,
            verbose=False,
            warm_start=warm_start,
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

    def _df_exclude_model(self, df, exclude: List[str], key="Model"):
        include_idx = np.where([all([y not in x for y in exclude]) for x in df[key]])
        df = df.loc[df.index[include_idx], :].reset_index(drop=True)
        return df

    def improvement(
        self, leaderboard: pd.DataFrame, cv_path=None, exclude: List[str] = None
    ):
        if exclude is not None:
            leaderboard = self._df_exclude_model(leaderboard, exclude)
        base = leaderboard[leaderboard["Program"] != self.program].reset_index(
            drop=True
        )
        improved = leaderboard[leaderboard["Program"] == self.program].reset_index(
            drop=True
        )
        improved_measure = improved[["Program", "Model"]].copy()
        if cv_path is not None:
            test_res = self.improvement_cv_test(
                model_names=list(improved["Model"]),
                metrics=list(base.columns[2:]),
                cv_path=cv_path,
            )
        else:
            test_res = None
        for idx, model in enumerate(improved["Model"]):
            components = model.split("_")
            modelbase = components[0]
            base_model = components[1]
            base_metrics = base[
                (base["Program"] == modelbase) & (base["Model"] == base_model)
            ]
            if len(base_metrics) > 1:
                raise Exception(
                    f"Conflict model {base_model} in model base {modelbase}."
                )
            improved_metrics = improved[improved["Model"] == model]
            if len(improved_metrics) > 1:
                raise Exception(f"Conflict model {model} in model base {self.program}.")
            for metric in base_metrics.columns[2:]:
                improved_metric = improved_metrics.loc[idx, metric]
                base_metric = base_metrics.squeeze()[metric]
                higher_better = 1 if any([m in metric for m in ["R2"]]) else -1
                improved_measure.loc[idx, f"{metric} Improvement"] = higher_better * (
                    improved_metric - base_metric
                )
                improved_measure.loc[idx, f"{metric} % Improvement"] = (
                    higher_better * (improved_metric - base_metric) / base_metric
                ) * 100
                if test_res is not None:
                    improved_measure.loc[idx, f"{metric} Improvement p-value"] = (
                        test_res[model][metric]["p-value"]
                    )
        improved_measure.sort_values(
            by="Testing RMSE % Improvement",
            ascending=False,
            inplace=True,
            ignore_index=True,
        )
        if test_res is not None:
            return improved_measure, test_res
        else:
            return improved_measure

    def method_ranking(self, improved_measure, leaderboard, exclude: List[str] = None):
        if exclude is not None:
            leaderboard = self._df_exclude_model(leaderboard, exclude)
            improved_measure = self._df_exclude_model(improved_measure, exclude)
        ranked_improvement = improved_measure.sort_values(
            by="Testing RMSE % Improvement", ascending=False, ignore_index=True
        )
        ranked_improvement["Rank"] = np.arange(1, len(ranked_improvement) + 1)
        imp_model_names = ranked_improvement["Model"].values.flatten()
        imp_methods_per_model = [name.split("_")[2:] for name in imp_model_names]
        leaderboard = leaderboard.copy()
        leaderboard["Rank"] = np.arange(1, len(leaderboard) + 1)
        lead_model_names = leaderboard["Model"].values.flatten()
        lead_methods_per_model = [
            name.split("_")[2:] if len(name.split("_")) > 2 else name
            for name in lead_model_names
        ]
        unique_methods = [np.unique(x) for x in zip(*imp_methods_per_model)]
        base_df = pd.DataFrame(
            columns=[
                "Method set",
                "Avg. % improvement ranking",
                "% improvement ranking p-value versus others",
                "Avg. leaderboard ranking",
                "leaderboard ranking p-value versus others",
            ],
            index=[],
        )
        dfs = []
        detailed = {}
        for method_set in unique_methods:
            if len(method_set) == 1:
                continue
            df = base_df.copy()
            for method in method_set:
                df.loc[method] = np.nan
            category = "/".join(method_set)
            df["Category"] = category
            detailed[category] = {}
            imp_ranks_method_set = {}
            lead_ranks_method_set = {}
            for method in method_set:
                imp_ranks = ranked_improvement["Rank"].values.flatten()[
                    np.where([method in methods for methods in imp_methods_per_model])[
                        0
                    ]
                ] / len(ranked_improvement)
                lead_ranks = leaderboard["Rank"].values.flatten()[
                    np.where([method in methods for methods in lead_methods_per_model])[
                        0
                    ]
                ] / len(leaderboard)
                imp_ranks_method_set[method] = imp_ranks
                lead_ranks_method_set[method] = lead_ranks
                df.loc[method, "Avg. % improvement ranking"] = np.mean(imp_ranks)
                df.loc[method, "Avg. leaderboard ranking"] = np.mean(lead_ranks)
            detailed[category]["imp"] = imp_ranks_method_set
            detailed[category]["lead"] = lead_ranks_method_set
            for method in method_set:
                s_imp = ""
                s_lead = ""
                for other_method in np.setdiff1d(method_set, [method]):
                    p_imp = getattr(
                        stats.ttest_ind(
                            imp_ranks_method_set[method],
                            imp_ranks_method_set[other_method],
                        ),
                        "pvalue",
                    )
                    p_lead = getattr(
                        stats.ttest_ind(
                            lead_ranks_method_set[method],
                            lead_ranks_method_set[other_method],
                        ),
                        "pvalue",
                    )
                    s_imp += f"{other_method}: {p_imp}; "
                    s_lead += f"{other_method}: {p_lead}; "
                df.loc[method, "% improvement ranking p-value versus others"] = s_imp
                df.loc[method, "leaderboard ranking p-value versus others"] = s_lead
            dfs.append(df)
        method_ranking = pd.concat(dfs, axis=0)
        return method_ranking, detailed

    def plot_method_ranking(
        self,
        method_ranking,
        save_to=None,
        palette=None,
        catplot_kwargs=None,
    ):
        _catplot_kwargs = update_defaults_by_kwargs(
            dict(
                legend_out=True,
                sharex=False,
                sharey=False,
                palette=palette,
                flierprops={"marker": "o"},
                fliersize=2,
                dodge=False,
                height=2,
                aspect=1,
            ),
            catplot_kwargs,
        )
        dfs = []
        title_dict = {"% Improvement ranking": {}, "Leaderboard ranking": {}}
        for idx, category in enumerate(method_ranking.keys()):
            df_dict = {"variable": [], "value": []}
            for method, ranks in method_ranking[category]["imp"].items():
                df_dict["variable"].extend([method] * len(ranks))
                df_dict["value"].extend(list(ranks))
            df_imp = pd.DataFrame(df_dict)
            df_imp["col"] = category
            df_imp["row"] = "% Improvement ranking"
            dfs.append(df_imp)
            title_dict["% Improvement ranking"][category] = category
            df_dict = {"variable": [], "value": []}
            for method, ranks in method_ranking[category]["lead"].items():
                df_dict["variable"].extend([method] * len(ranks))
                df_dict["value"].extend(list(ranks))
            df_lead = pd.DataFrame(df_dict)
            df_lead["col"] = category
            df_lead["row"] = "Leaderboard ranking"
            dfs.append(df_lead)
            title_dict["Leaderboard ranking"][category] = ""
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=DeprecationWarning)
            g = sns.catplot(
                kind="box",
                data=pd.concat(dfs, ignore_index=True),
                col="col",
                row="row",
                x="variable",
                y="value",
                hue="variable",
                **_catplot_kwargs,
            )
        # g.add_legend(**_legend_kwargs)
        # g.set_titles(row_template="{row_name}", col_template="{col_name}")
        plt.setp(g.axes, xlabel="")
        g.axes[0, 0].set_ylabel("\% Improvement ranking")
        g.axes[1, 0].set_ylabel("Leaderboard ranking")
        for (row_key, col_key), ax in g.axes_dict.items():
            ax.set_title(title_dict[row_key][col_key], fontsize=10)
        plt.tight_layout()
        if save_to is not None:
            plt.savefig(save_to, dpi=500)
        plt.show()
        plt.close()

    def plot_compare(
        self,
        leaderboard,
        improved_measure,
        test_res,
        metric,
        p_symbol_map_func,
        clr=None,
        ax=None,
        p_cap_width=0.4,
        p_cap_height=0.05,
        p_pos_y=0.3,
        p_text_up_pos_y=0.28,
        p_text_kwargs=None,
        save_show_close=True,
        figure_kwargs=None,
        boxplot_kwargs=None,
        legend_kwargs=None,
        savefig_kwargs=None,
    ):
        leaderboard = leaderboard[leaderboard["Program"] != self.program]
        model_names = improved_measure["Model"]
        clr = global_palette if clr is None else clr
        legend_kwargs_ = update_defaults_by_kwargs(
            dict(title=None, frameon=False), legend_kwargs
        )
        figure_kwargs_ = update_defaults_by_kwargs(dict(), figure_kwargs)
        boxplot_kwargs_ = update_defaults_by_kwargs(
            dict(
                orient="h",
                linewidth=1,
                fliersize=2,
                flierprops={"marker": "o"},
                palette=clr,
                saturation=1,
            ),
            boxplot_kwargs,
        )
        p_text_kwargs_ = update_defaults_by_kwargs(dict(), p_text_kwargs)
        ax, given_ax = self.trainer._plot_action_init_ax(ax, figure_kwargs_)
        orient = boxplot_kwargs_["orient"]
        dfs = []
        p_values = {}
        for idx, (program, model) in enumerate(
            zip(leaderboard["Program"], leaderboard["Model"])
        ):
            improve_models = [m for m in model_names if f"{program}_{model}" in m]
            if len(improve_models) == 0:
                continue
            base_metrics = {
                "Base model": test_res[improve_models[0]][metric]["base"],
            }
            improved_metrics = {
                "-".join(m.split("_")[2:]): test_res[m][metric]["improved"]
                for m in improve_models
            }
            if len(improve_models) == 1:
                p_values[idx] = test_res[improve_models[0]][metric]["p-value"]
            improved_metrics = {
                key: improved_metrics[key] for key in sorted(improved_metrics.keys())
            }
            base_metrics.update(improved_metrics)
            df = pd.DataFrame(base_metrics).melt()
            df["class"] = f"{program}-{model}"
            df["hue"] = [x if x == "Base model" else "Proposed" for x in df["variable"]]
            dfs.append(df)
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=DeprecationWarning)
            sns.boxplot(
                data=pd.concat(dfs, ignore_index=True),
                x="value" if orient == "h" else "class",
                y="class" if orient == "h" else "value",
                hue="hue",  # "variable"
                ax=ax,
                **boxplot_kwargs_,
            )
        ax.get_legend().remove()
        ax.legend(**legend_kwargs_)
        for idx, p_val in p_values.items():
            x1, x2 = idx - p_cap_width / 2, idx + p_cap_width / 2
            y1, y2 = p_pos_y + p_cap_height, p_pos_y
            ax.plot([x1, x1, x2, x2], [y1, y2, y2, y1], lw=0.5, c="k")
            ax.text(
                (x1 + x2) / 2,
                p_text_up_pos_y,
                p_symbol_map_func(p_val),
                ha="center",
                va="top",
                **p_text_kwargs_,
            )

        return self.trainer._plot_action_after_plot(
            fig_name=os.path.join(self.trainer.project_root, f"compare.pdf"),
            disable=given_ax,
            ax_or_fig=ax,
            xlabel=metric if orient == "h" else None,
            ylabel=metric if orient == "v" else None,
            tight_layout=True,
            save_show_close=save_show_close,
            savefig_kwargs=savefig_kwargs,
        )

    def plot_improvement(
        self,
        leaderboard,
        improved_measure,
        test_res,
        metric,
        clr=None,
        save_show_close=True,
        ax=None,
        figure_kwargs=None,
        barplot_kwargs=None,
        legend_kwargs=None,
        savefig_kwargs=None,
    ):
        leaderboard = leaderboard[leaderboard["Program"] != self.program]
        model_names = improved_measure["Model"]
        clr = global_palette if clr is None else clr
        figure_kwargs_ = update_defaults_by_kwargs(dict(), figure_kwargs)
        legend_kwargs_ = update_defaults_by_kwargs(
            dict(title=None, frameon=False), legend_kwargs
        )
        barplot_kwargs_ = update_defaults_by_kwargs(
            dict(
                orient="h",
                linewidth=1,
                edgecolor="k",
                saturation=1,
                palette=clr,
            ),
            barplot_kwargs,
        )
        ax, given_ax = self.trainer._plot_action_init_ax(ax, figure_kwargs_)
        orient = barplot_kwargs_["orient"]
        dfs = []
        if type(metric) == str:
            metric = [metric]
        for idx, (program, model) in enumerate(
            zip(leaderboard["Program"], leaderboard["Model"])
        ):
            improve_models = [m for m in model_names if f"{program}_{model}" in m]
            if len(improve_models) == 0:
                continue
            for met in metric:
                improved_metrics = {
                    "-".join(m.split("_")[2:]): (
                        test_res[m][met]["base"] - test_res[m][met]["improved"]
                    )
                    / test_res[m][met]["base"]
                    * 100
                    for m in improve_models
                }
                improved_metrics = {
                    key: improved_metrics[key]
                    for key in sorted(improved_metrics.keys())
                }
                df = pd.DataFrame(improved_metrics).melt()
                df["class"] = f"{program}-{model}"
                df["hue"] = [met] * len(df)
                dfs.append(df)
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=DeprecationWarning)
            sns.barplot(
                data=pd.concat(dfs, ignore_index=True),
                x="value" if orient == "h" else "class",
                y="class" if orient == "h" else "value",
                hue="hue" if len(metric) > 1 else None,
                ax=ax,
                **barplot_kwargs_,
            )
        ax.get_legend().remove()
        ax.legend(**legend_kwargs_)

        return self.trainer._plot_action_after_plot(
            fig_name=os.path.join(self.trainer.project_root, f"improvement.pdf"),
            disable=given_ax,
            ax_or_fig=ax,
            xlabel=(
                f"Improvement of {metric[0] if len(metric) == 1 else 'metrics'} (Percentage)"
                if orient == "h"
                else None
            ),
            ylabel=(
                f"Improvement of {metric[0] if len(metric) == 1 else 'metrics'} (Percentage)"
                if orient == "v"
                else None
            ),
            tight_layout=True,
            save_show_close=save_show_close,
            savefig_kwargs=savefig_kwargs,
        )

    def improvement_cv_test(
        self,
        model_names: Union[List[str], str],
        metrics: Union[List[str], str],
        cv_path,
    ):
        if type(model_names) == str:
            model_names = [model_names]
        if type(metrics) == str:
            metrics = [metrics]
        cv_names = [path for path in os.listdir(cv_path) if path.endswith(".csv")]
        res = {
            name: {
                metric: {
                    "base": np.array([]),
                    "improved": np.array([]),
                    "p-value": None,
                }
                for metric in metrics
            }
            for name in model_names
        }
        for cv_name in cv_names:
            df = pd.read_csv(os.path.join(cv_path, cv_name), index_col=0)
            for model_name in model_names:
                components = model_name.split("_")
                modelbase = components[0]
                base_model = components[1]
                base_metrics = df[
                    (df["Program"] == modelbase) & (df["Model"] == base_model)
                ].squeeze()
                improved_metrics = df[
                    (df["Program"] == self.program) & (df["Model"] == model_name)
                ].squeeze()
                for metric in metrics:
                    base_metric = base_metrics[metric]
                    improved_metric = improved_metrics[metric]
                    res[model_name][metric]["base"] = np.append(
                        res[model_name][metric]["base"], base_metric
                    )
                    res[model_name][metric]["improved"] = np.append(
                        res[model_name][metric]["improved"], improved_metric
                    )
        for model_name in model_names:
            for metric in metrics:
                res[model_name][metric]["p-value"] = stats.mannwhitneyu(
                    res[model_name][metric]["base"],
                    res[model_name][metric]["improved"],
                )[1]
        return res

    def inspect_weighted_predictions(self, model_name, **kwargs):
        model = self.model[model_name]
        target_attr = ["dl_weight", "dl_pred", "phy_pred"]
        if not hasattr(model, "dl_weight"):
            raise Exception(
                f"The model does not have the attribute `dl_weight`. Is it trained?"
            )
        inspect_dict = self.inspect_attr(
            model_name=model_name, attributes=target_attr, **kwargs
        )
        return inspect_dict

    def inspect_phy_models(self, model_name, **kwargs):
        target_attr = ["clustering_phy_model"]
        inspect_dict = self.inspect_attr(
            model_name=model_name, attributes=target_attr, to_numpy=False, **kwargs
        )
        phys = inspect_dict["train"]["clustering_phy_model"].phys
        phy_weight = (
            inspect_dict["train"]["clustering_phy_model"]
            .running_phy_weight.data.detach()
            .cpu()
        )
        norm_phy_weight = nn.functional.normalize(torch.abs(phy_weight), p=1).numpy()
        return phys, norm_phy_weight

    def inspect_clusters(self, model_name, **kwargs):
        target_attr = ["clustering_phy_model"]
        inspect_dict = self.inspect_attr(
            model_name=model_name, attributes=target_attr, **kwargs
        )
        to_cpu = lambda x: x.detach().cpu().numpy()
        if "USER_INPUT" in inspect_dict.keys():
            return to_cpu(inspect_dict["USER_INPUT"]["clustering_phy_model"].x_cluster)
        else:
            cluster_train = to_cpu(
                inspect_dict["train"]["clustering_phy_model"].x_cluster
            )
            cluster_val = to_cpu(inspect_dict["val"]["clustering_phy_model"].x_cluster)
            cluster_test = to_cpu(
                inspect_dict["test"]["clustering_phy_model"].x_cluster
            )
            return cluster_train, cluster_val, cluster_test

    def df_with_cluster(self, model_name, save_to: str = None, **kwargs):
        res = self.inspect_clusters(
            model_name,
            df=self.trainer.df,
            derived_data=self.trainer.derived_data,
            **kwargs,
        )
        df = self.trainer.df.copy()
        df["cluster"] = res
        if save_to is not None:
            if not save_to.endswith(".csv"):
                raise Exception(f"Can only save to a csv file.")
            df.to_csv(save_to, index=False)
        return df

    def plot_phy_weights(self, model_name, save_to=None, **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        phys, phy_weight = self.inspect_phy_models(model_name=model_name, **kwargs)
        names = [phy.__class__.__name__ for phy in phys]
        fig = plt.figure(figsize=(10, 4))
        gs = GridSpec(100, 100, figure=fig)
        ax = fig.add_subplot(gs[:97, 10:97])
        im = ax.imshow(phy_weight.T, cmap="Blues")
        ax.set_yticklabels(names)
        ax.set_yticks(np.arange(phy_weight.shape[1]))
        ax.set_xticks(np.arange(phy_weight.shape[0]))
        ax.set_xlabel("ID of clusters")
        ax.set_title("Weights of physical models")
        cax = fig.add_subplot(gs[50:78, 98:])
        plt.colorbar(mappable=im, cax=cax)
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
