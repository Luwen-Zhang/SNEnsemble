from src.utils import *
from src.model import TorchModel
from ._transformer.models_clustering import *
from ._transformer.models_with_seq import *
from ._transformer.models_basic import *
from itertools import product
from scipy import stats
from skopt.space import Integer


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
                ["NoWrap", "Wrap"],
                ["1L", "2L"],
                ["PCA"],
                ["KMeans", "GMM", "BMM"],
            )
        ]
        return all_names

    def _new_model(self, model_name, verbose, required_models=None, **kwargs):
        database = self.trainer.args["database"]
        if "composite" in database:
            sn_category = "composite"
        elif "alloy" in database:
            sn_category = "alloy"
        else:
            warnings.warn(
                f"Neither `composite` nor `alloy` is in the name of the database `{database}`. "
                f"Only parts of the SN models are used."
            )
            sn_category = None
        fix_kwargs = dict(
            n_inputs=len(self.datamodule.cont_feature_names),
            n_outputs=len(self.datamodule.label_name),
            layers=self.datamodule.args["layers"],
            cat_num_unique=[len(x) for x in self.trainer.cat_feature_mapping.values()],
            datamodule=self.datamodule,
            sn_category=sn_category,
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

    def _custom_training_params(self, model_name) -> Dict:
        return dict(epoch=50, bayes_calls=20, bayes_epoch=5)

    def _initial_values(self, model_name):
        components = model_name.split("_")
        res = {}
        if "1L" in components:
            res = {"n_clusters": 32}
        elif "2L" in components:
            res = {"n_clusters": 32, "n_clusters_per_cluster": 8}
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
        if "2L" in components:
            datamodule = getattr(self, "datamodule", self.trainer.datamodule)
            top_level_clustering_features = (
                AbstractClusteringModel.top_clustering_features_idx(datamodule)
            )
            if len(top_level_clustering_features) == 0:
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

    def improvement(self, leaderboard: pd.DataFrame, cv_path=None):
        base = leaderboard[leaderboard["Program"] != self.program].reset_index(
            drop=True
        )
        improved = leaderboard[leaderboard["Program"] == self.program].reset_index(
            drop=True
        )
        improved_measure = improved[["Program", "Model"]].copy()
        if cv_path is not None:
            ttest_res = self.improvement_cv_ttest(
                model_names=list(improved["Model"]),
                metrics=list(base.columns[2:]),
                cv_path=cv_path,
            )
        else:
            ttest_res = None
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
                if ttest_res is not None:
                    improved_measure.loc[
                        idx, f"{metric} Improvement p-value"
                    ] = ttest_res[model][metric]["p-value"]
        improved_measure.sort_values(
            by="Testing RMSE % Improvement",
            ascending=False,
            inplace=True,
            ignore_index=True,
        )
        if ttest_res is not None:
            return improved_measure, ttest_res
        else:
            return improved_measure

    def method_ranking(self, improved_measure, leaderboard):
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
        _catplot_kwargs = dict(
            legend_out=True,
            sharex=False,
            sharey=False,
            palette=palette,
            flierprops={"marker": "o"},
            fliersize=2,
            dodge=False,
            height=2,
            aspect=1,
        )
        if catplot_kwargs is not None:
            _catplot_kwargs.update(catplot_kwargs)
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

    def plot_improvement(
        self,
        leaderboard,
        improved_measure,
        ttest_res,
        metric,
        save_to=None,
        palette=None,
        figsize_kwargs=None,
        catplot_kwargs=None,
        legend_kwargs=None,
        adjust_kwargs=None,
    ):
        leaderboard = leaderboard[leaderboard["Program"] != self.program]
        model_names = improved_measure["Model"]
        if palette is None:
            palette = sns.color_palette(global_palette)
        _figsize_kwargs = dict(
            max_col=5, width_per_item=1.6, height_per_item=1.6, max_width=5
        )
        if figsize_kwargs is not None:
            _figsize_kwargs.update(figsize_kwargs)
        figsize, width, height = get_figsize(n=len(leaderboard), **_figsize_kwargs)
        _legend_kwargs = dict(bbox_to_anchor=(0.85, 0.075), ncol=4)
        if legend_kwargs is not None:
            _legend_kwargs.update(legend_kwargs)
        _adjust_kwargs = dict(bottom=0.15)
        if adjust_kwargs is not None:
            _adjust_kwargs.update(adjust_kwargs)
        _catplot_kwargs = dict(
            legend_out=True,
            sharex=False,
            sharey=False,
            palette=palette,
            flierprops={"marker": "o"},
            fliersize=2,
            dodge=False,
            height=figsize[1] / height,
            aspect=0.4,
        )
        if catplot_kwargs is not None:
            _catplot_kwargs.update(catplot_kwargs)
        dfs = []
        title_dict = {row: {} for row in range(height)}
        for idx, (program, model) in enumerate(
            zip(leaderboard["Program"], leaderboard["Model"])
        ):
            improve_models = [m for m in model_names if f"{program}_{model}" in m]
            base_metrics = {
                "Base model": ttest_res[improve_models[0]][metric]["base"],
            }
            improved_metrics = {
                "-".join(m.split("_")[2:]): ttest_res[m][metric]["improved"]
                for m in improve_models
            }
            improved_metrics = {
                key: improved_metrics[key] for key in sorted(improved_metrics.keys())
            }
            base_metrics.update(improved_metrics)
            # Remove outliers that are too far away (greater than Q3+3IQR) for better plotting.
            for key in base_metrics.keys():
                vals = base_metrics[key]
                q1 = np.quantile(vals, 0.25)
                q3 = np.quantile(vals, 0.75)
                iqr = q3 - q1
                vals[vals > q3 + 3 * iqr] = np.nan
                base_metrics[key] = vals
            df = pd.DataFrame(base_metrics).melt()
            col = idx % width
            row = idx // width
            df["col"] = col
            df["row"] = row
            title_dict[row][col] = f"{program}\n{model}"
            dfs.append(df)
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
        g.add_legend(**_legend_kwargs)
        for (row_key, col_key), ax in g.axes_dict.items():
            ax.set_title(title_dict[row_key][col_key], fontsize=10)
        plt.setp(g.axes, xticks=[], xlabel="", ylabel="")
        plt.tight_layout()
        plt.subplots_adjust(**_adjust_kwargs)
        if save_to is not None:
            plt.savefig(save_to, dpi=500)
        plt.show()
        plt.close()

    def improvement_cv_ttest(
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
                res[model_name][metric]["p-value"] = getattr(
                    stats.ttest_ind(
                        res[model_name][metric]["base"],
                        res[model_name][metric]["improved"],
                    ),
                    "pvalue",
                )
        return res

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
        return inspect_dict

    def inspect_phy_models(self, model_name, **kwargs):
        target_attr = ["clustering_sn_model"]
        inspect_dict = self.inspect_attr(
            model_name=model_name, attributes=target_attr, to_numpy=False, **kwargs
        )
        sns = inspect_dict["train"]["clustering_sn_model"].sns
        sn_weight = (
            inspect_dict["train"]["clustering_sn_model"]
            .running_sn_weight.data.detach()
            .cpu()
        )
        norm_sn_weight = nn.functional.normalize(torch.abs(sn_weight), p=1).numpy()
        return sns, norm_sn_weight

    def inspect_clusters(self, model_name, **kwargs):
        target_attr = ["clustering_sn_model"]
        inspect_dict = self.inspect_attr(
            model_name=model_name, attributes=target_attr, **kwargs
        )
        to_cpu = lambda x: x.detach().cpu().numpy()
        if "USER_INPUT" in inspect_dict.keys():
            return to_cpu(inspect_dict["USER_INPUT"]["clustering_sn_model"].x_cluster)
        else:
            cluster_train = to_cpu(
                inspect_dict["train"]["clustering_sn_model"].x_cluster
            )
            cluster_val = to_cpu(inspect_dict["val"]["clustering_sn_model"].x_cluster)
            cluster_test = to_cpu(inspect_dict["test"]["clustering_sn_model"].x_cluster)
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

        sns, sn_weight = self.inspect_phy_models(model_name=model_name, **kwargs)
        names = [sn.__class__.__name__ for sn in sns]
        fig = plt.figure(figsize=(8, 3))
        gs = GridSpec(100, 100, figure=fig)
        ax = fig.add_subplot(gs[:97, 10:97])
        im = ax.imshow(sn_weight.T, cmap="Blues")
        ax.set_yticklabels(names)
        ax.set_yticks(np.arange(sn_weight.shape[1]))
        ax.set_xticks(np.arange(sn_weight.shape[0]))
        ax.set_xlabel("ID of clusters")
        ax.set_title("Weights of physical models")
        cax = fig.add_subplot(gs[50:96, 98:])
        plt.colorbar(mappable=im, cax=cax)
        plt.tight_layout()
        if save_to is not None:
            plt.savefig(save_to, dpi=500)
        plt.show()
        plt.close()

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

        if "USER_INPUT" in inspect_dict.keys():
            fig = plt.figure(figsize=(4, 4))
            ax = plt.subplot(111)
            plot_part(inspect_dict["USER_INPUT"], "Investigated set", ax)
        else:
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
