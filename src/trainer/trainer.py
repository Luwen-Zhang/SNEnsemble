from tabensemb.trainer import Trainer
from tabensemb.utils import *
from typing import List, Tuple
import scipy.stats as st
import pandas as pd


class FatigueTrainer(Trainer):
    def get_material_code(
        self, unique: bool = False, partition: str = "all"
    ) -> pd.DataFrame:
        """
        Get Material_Code of the dataset.
        Parameters
        ----------
        unique
            If True, values in the Material_Code column will be counted.
        partition
            "train", "val", "test", or "all". See ``Trainer._get_indices``.
        Returns
        -------
        m_code
            If unique is True, the returned dataframe contains counts for each material code in the selected partition.
            Otherwise, the original material codes in the selected partition are returned.
        """
        material_code = self.df[["Material_Code"]]
        indices = self.datamodule._get_indices(partition=partition)
        if unique:
            unique_list = list(sorted(set(material_code.loc[indices, "Material_Code"])))
            val_cnt = material_code.loc[indices, :].value_counts()
            return pd.DataFrame(
                {
                    "Material_Code": unique_list,
                    "Count": [val_cnt[x] for x in unique_list],
                }
            )
        else:
            return material_code.loc[indices, :]

    def select_by_material_code(
        self, m_code: str, partition: str = "all", select_by_value_kwargs: Dict = None
    ):
        """
        Select samples with the specified material code.

        Parameters
        ----------
        m_code
            The selected material code.
        partition
            "train", "val", "test", or "all". See ``Trainer._get_indices``.

        Returns
        -------
        indices
            The pandas index where the material code exists.
        """
        select_by_value_kwargs_ = update_defaults_by_kwargs(
            dict(partition=partition), select_by_value_kwargs
        )
        if "selection" in select_by_value_kwargs_.keys():
            select_by_value_kwargs_["selection"].update({"Material_Code": m_code})
        else:
            select_by_value_kwargs_["selection"] = {"Material_Code": m_code}
        return self.datamodule.select_by_value(**select_by_value_kwargs_)

    def cal_theoretical_pof50(
        self,
        max_stress_col="Maximum Stress",
        r_value_col="R-value",
        freq_col="Frequency",
        distribution="gaussian",
    ):
        from ..data.dataderiver import TheoreticalFiftyPofDeriver

        deriver = TheoreticalFiftyPofDeriver(
            derived_name="_",
            max_stress_col=max_stress_col,
            r_value_col=r_value_col,
            freq_col=freq_col,
            distribution=distribution,
        )
        pof50, _ = deriver.derive(self.df, self.datamodule)
        deriver.describe_acc(
            self.df[self.label_name].values, pof50, self.datamodule, distribution
        )
        return pof50

    def plot_data_split(
        self,
        bins: int = 30,
        percentile: str = "all",
        fig=None,
        savefig_kwargs: Dict = None,
        save_show_close: bool = True,
        figure_kwargs: Dict = None,
    ):
        """
        Visualize how the dataset is split into training/validation/testing datasets.

        Parameters
        ----------
        bins
            Number of bins to divide the target value of each material.
        percentile
            If "all", the limits of x-axis will be the overall range of the target. Otherwise, the limits will be [0,1]
            representing the individual percentile of target for each material.
        """
        from matplotlib.gridspec import GridSpec

        figure_kwargs_ = update_defaults_by_kwargs(
            dict(figsize=(8, 2.5)), figure_kwargs
        )

        train_m_code = list(
            self.get_material_code(unique=True, partition="train")["Material_Code"]
        )
        val_m_code = list(
            self.get_material_code(unique=True, partition="val")["Material_Code"]
        )
        test_m_code = list(
            self.get_material_code(unique=True, partition="test")["Material_Code"]
        )
        all_m_code = list(self.get_material_code(unique=True)["Material_Code"])
        no_train_m_code = np.setdiff1d(all_m_code, train_m_code)
        val_only_m_code = np.setdiff1d(no_train_m_code, test_m_code)
        rest_m_code = np.setdiff1d(no_train_m_code, val_only_m_code)

        all_m_code = train_m_code + list(val_only_m_code) + list(rest_m_code)

        all_cycle = self.df[self.label_name[0]].values.flatten()
        length = len(all_m_code)
        train_heat = np.zeros((length, bins))
        val_heat = np.zeros((length, bins))
        test_heat = np.zeros((length, bins))
        partition_cycles = {"train": [], "val": [], "test": []}
        np.seterr(invalid="ignore")
        for idx, material in enumerate(all_m_code):
            cycle = all_cycle[
                self.select_by_material_code(m_code=material, partition="all")
            ]
            if percentile == "all":
                hist_range = (np.min(all_cycle), np.max(all_cycle))
            else:
                hist_range = (np.min(cycle), np.max(cycle))

            def get_heat(partition):
                cycles = all_cycle[
                    self.select_by_material_code(m_code=material, partition=partition)
                ]
                partition_cycles[partition].append(cycles)
                if len(cycles) <= 1:
                    return np.zeros(bins)
                else:
                    res = st.gaussian_kde(cycles)(
                        np.linspace(hist_range[0], hist_range[1], bins)
                    )
                    return res / np.max(res)

            train_heat[idx, :] = get_heat(partition="train")
            val_heat[idx, :] = get_heat(partition="val")
            test_heat[idx, :] = get_heat(partition="test")

        train_heat[np.isnan(train_heat)] = 0
        val_heat[np.isnan(val_heat)] = 0
        test_heat[np.isnan(test_heat)] = 0
        fig, given_ax = self._plot_action_init_ax(fig, figure_kwargs_, return_fig=True)

        gs = GridSpec(100, 100, figure=fig)

        def plot_im(heat, pos, hide_y_ticks=False):
            ax = fig.add_subplot(pos)
            im = ax.imshow(heat, aspect="auto", cmap="Oranges")

            ax.set_ylim([0, length])
            ax.set_yticks([0, length])
            if hide_y_ticks:
                ax.set_yticklabels([])
            ax.set_xlim([-0.5, bins - 0.5])
            ax.set_xticks([0 - 0.5, (bins - 1) / 2, bins - 0.5])
            if percentile == "all":
                ax.set_xticklabels(
                    [
                        f"{x:.1f}"
                        for x in [hist_range[0], np.mean(hist_range), hist_range[1]]
                    ]
                )
            else:
                ax.set_xticklabels([0, 50, 100])
            return im

        def plot_kde(heat, pos, name, hide_y_ticks=False):
            ax = fig.add_subplot(pos)
            x = np.linspace(np.min(all_cycle), np.max(all_cycle), 100)
            kde = st.gaussian_kde(np.hstack(heat))(x)
            ax.plot(x, kde, c=plt.get_cmap("Oranges")(200), lw=1)
            mean_heat = np.mean(np.hstack(heat))
            ax.plot(
                np.array([mean_heat] * 10),
                np.linspace(0, 0.5, 10),
                "--",
                color="grey",
                alpha=0.5,
                lw=1,
            )
            ax.set_ylim([0, np.max(kde) * 1.5])
            ax.set_yticks([0, np.max(kde) * 1.5])
            ax.set_yticklabels(
                ["0", str(round(np.max(kde) * 1.5, 2))],
                fontsize=plt.rcParams["font.size"] * 0.8,
            )
            ax.text(
                mean_heat + 0.13,
                ax.get_ylim()[1] * 0.94,
                f"Mean: {mean_heat:.2f}",
                fontsize=plt.rcParams["font.size"] * 0.8,
                ha="left",
                va="top",
            )
            if hide_y_ticks:
                ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_xlim([np.min(all_cycle), np.max(all_cycle)])
            ax.set_title(name)

        plot_kde(
            partition_cycles["train"],
            gs[:17, 0:29],
            name="Training set",
            hide_y_ticks=False,
        )
        plot_kde(
            partition_cycles["val"],
            gs[:17, 34:62],
            name="Validation set",
            hide_y_ticks=False,
        )
        plot_kde(
            partition_cycles["test"],
            gs[:17, 67:95],
            name="Testing set",
            hide_y_ticks=False,
        )
        plot_im(train_heat, gs[26:, 0:29], hide_y_ticks=False)
        plot_im(val_heat, gs[26:, 34:62], hide_y_ticks=True)
        im = plot_im(test_heat, gs[26:, 67:95], hide_y_ticks=True)
        # plt.colorbar(mappable=im)
        cax = fig.add_subplot(gs[50:98, 97:])
        cbar = plt.colorbar(cax=cax, mappable=im)
        cax.set_ylabel("Normalized density")
        ax = fig.add_subplot(gs[:20, :], frameon=False)
        plt.tick_params(
            labelcolor="none",
            which="both",
            top=False,
            bottom=False,
            left=False,
            right=False,
        )
        ax.set_ylabel(f"Density\n")
        ax = fig.add_subplot(gs[25:, :], frameon=False)
        plt.tick_params(
            labelcolor="none",
            which="both",
            top=False,
            bottom=False,
            left=False,
            right=False,
        )
        self._plot_action_after_plot(
            fig_name=os.path.join(
                self.project_root,
                f"{self.datamodule.datasplitter.__class__.__name__}_{percentile}.pdf",
            ),
            disable=False,
            ax_or_fig=ax,
            xlabel=(
                self.label_name[0]
                if percentile == "all"
                else f"Percentile of {self.label_name[0]} for each material"
            ),
            ylabel=f"No. of Material",
            tight_layout=True,
            save_show_close=save_show_close,
            savefig_kwargs=savefig_kwargs,
        )
        return fig

    def plot_multiple_S_N(
        self, m_codes: List[str], hide_plt_show: bool = True, **kwargs
    ):
        """
        A utility function to call ``Trainer.plot_S_N`` for multiple times with the same settings.

        Parameters
        ----------
        m_codes
            A list of material codes.
        hide_plt_show
            Whether to prevent the matplotlib figure showing in canvas.
        kwargs
            Arguments for ``Trainer.plot_S_N``.
        """
        for m_code in m_codes:
            print(m_code)
            if hide_plt_show:
                with HiddenPltShow():
                    self.plot_S_N(m_code=m_code, **kwargs)
            else:
                self.plot_S_N(m_code=m_code, **kwargs)

    def plot_S_N(
        self,
        s_col: str,
        n_col: str,
        m_code: str,
        CI: float = 0.95,
        method: str = "statistical",
        verbose: bool = True,
        program: str = "ThisWork",
        model_name: str = "ThisWork",
        refit: bool = True,
        log_stress: bool = False,
        plot_pof_area: bool = False,
        plot_scatter: bool = True,
        ax=None,
        train_val_test_color: List = None,
        select_by_value_kwargs: Dict = None,
        plot_kwargs: Dict = None,
        scatter_kwargs: Dict = None,
        legend_kwargs: Dict = None,
        figure_kwargs: Dict = None,
        savefig_kwargs: Dict = None,
        save_show_close: bool = True,
        **kwargs,
    ):
        """
        Calculate and plot the SN curve for the selected material using a selected model with bootstrap resampling.

        Parameters
        ----------
        s_col
            The name of the stress column.
        n_col
            The name of the log(fatigue life) column.
        ax
            A matplotlib Axis instance. If not None, a new figure will be initialized.
        CI
            The confidence interval for PSN curves and for predicted SN curves across multiple bootstrap runs and
            multiple samples. The latter usage is different from the CI argument in``Trainer.cal_partial_dependence``.
        method
            The method to calculate the confidence interval. See ``Trainer._sn_interval`` for details.
        verbose
            Verbosity.
        program
            The selected database.
        model_name
            The selected model in the database.
        refit
            Whether to refit models on bootstrapped datasets. See Trainer._bootstrap_fit.
        log_stress
            Whether to plot the stress in log scale.
        plot_pof_area
            Whether to calculate probability of failure shadows.
        **kwargs
            Other arguments for ``Trainer._bootstrap_fit``
        """
        figure_kwargs_ = update_defaults_by_kwargs(dict(), figure_kwargs)
        select_by_value_kwargs_ = update_defaults_by_kwargs(
            dict(), select_by_value_kwargs
        )
        legend_kwargs_ = update_defaults_by_kwargs(
            dict(
                loc="upper right",
                markerscale=1.5,
                handlelength=1,
                handleheight=0.9,
                fontsize=plt.rcParams["font.size"] * 0.8,
            ),
            legend_kwargs,
        )

        # Find the selected material.
        m_train_indices = self.select_by_material_code(
            m_code, partition="train", select_by_value_kwargs=select_by_value_kwargs_
        )
        m_test_indices = self.select_by_material_code(
            m_code, partition="test", select_by_value_kwargs=select_by_value_kwargs_
        )
        m_val_indices = self.select_by_material_code(
            m_code, partition="val", select_by_value_kwargs=select_by_value_kwargs_
        )
        original_indices = self.select_by_material_code(
            m_code, partition="all", select_by_value_kwargs=select_by_value_kwargs_
        )
        print(
            f"Train: {len(m_train_indices)}, val: {len(m_val_indices)}, test: {len(m_test_indices)}"
        )

        if len(original_indices) == 0:
            raise Exception(f"Selection not available.")

        # Extract S and N.
        s_train = self.df.loc[m_train_indices, s_col]
        n_train = self.df.loc[m_train_indices, n_col]
        s_val = self.df.loc[m_val_indices, s_col]
        n_val = self.df.loc[m_val_indices, n_col]
        s_test = self.df.loc[m_test_indices, s_col]
        n_test = self.df.loc[m_test_indices, n_col]

        all_s = np.vstack(
            [
                s_train.values.reshape(-1, 1),
                s_val.values.reshape(-1, 1),
                s_test.values.reshape(-1, 1),
            ]
        )

        # Determine the prediction range.
        s_min = np.min(all_s) - np.abs(np.max(all_s) - np.min(all_s)) * 0.5
        s_max = np.max(all_s) + np.abs(np.max(all_s) - np.min(all_s)) * 0.5

        # Get bootstrap predictions and confidence intervals from program-model_name
        chosen_indices = (
            m_train_indices
            if len(m_train_indices) != 0
            else np.append(m_val_indices, m_test_indices)
        )
        returned = self._bootstrap_fit(
            program=program,
            df=self.df.copy().loc[chosen_indices, :],
            derived_data=self.datamodule.get_derived_data_slice(
                self.derived_data, chosen_indices
            ),
            focus_feature=s_col,
            x_min=s_min,
            x_max=s_max,
            CI=CI,
            average=False,
            model_name=model_name,
            refit=refit if len(m_train_indices) != 0 else False,
            **kwargs,
        )
        if "inspect_attr_kwargs" in kwargs.keys():
            x_value, mean_pred, ci_left, ci_right, inspects = returned
        else:
            x_value, mean_pred, ci_left, ci_right = returned
            inspects = None

        # Defining a series of utilities.
        def get_interval_psn(s, n, xvals, n_pred_vals=None):
            # Calculate predictions, intervals, and psn from lin-log or log-log S and N.
            from sklearn.linear_model import LinearRegression

            lr = LinearRegression()
            lr.fit(s.reshape(-1, 1), n.reshape(-1, 1))
            n_pred_interp = (
                lr.predict(xvals.reshape(-1, 1)).flatten()
                if n_pred_vals is None
                else n_pred_vals
            )
            n_pred = (
                lr.predict(s.reshape(-1, 1))
                if n_pred_vals is None
                else np.interp(s, xvals, n_pred_vals).reshape(-1, 1)
            )
            CL, CR = self._sn_interval(
                method=method,
                y=n,
                y_pred=n_pred,
                x=s,
                xvals=xvals,
                CI=CI,
            )
            ci_left, ci_right = n_pred_interp - CL, n_pred_interp + CR
            psn_CL = self._psn(
                method="iso",
                y=n,
                y_pred=n_pred,
                x=s,
                xvals=xvals,
                CI=CI,
                p=0.95,
            )
            psn_pred = n_pred_interp - psn_CL
            return n_pred_interp, ci_left, ci_right, psn_pred

        def scatter_plot_func(x, y, color, name):
            # Plot training, validation, and testing sets.
            scatter_kwargs_ = update_defaults_by_kwargs(
                dict(
                    s=20,
                    color=color,
                    marker="o",
                    linewidth=0.4,
                    edgecolors="k",
                    label=f"{name} set",
                    zorder=20,
                ),
                scatter_kwargs,
            )
            ax.scatter(x, y, **scatter_kwargs_)

        def in_fill_between(x_arr, y_arr, xvals, cl, cr):
            # Calculate the number of points that are in the interval.
            def point_in_fill_between(x, y):
                which_x = np.where(np.abs(x - xvals) == np.min(np.abs(x - xvals)))[0][0]
                cl_x = cl[which_x]
                cr_x = cr[which_x]
                return True if cl_x <= y <= cr_x else False

            res = []
            for x, y in zip(x_arr, y_arr):
                res.append(point_in_fill_between(x, y))
            return np.count_nonzero(np.array(res))

        def report(interv_left, interv_right):
            # Report the number of points that are in the interval for three sets.
            print(
                f"Training {in_fill_between(s_train, n_train, x_value, interv_left, interv_right)}/{len(s_train)}"
            )
            print(
                f"Validation {in_fill_between(s_val, n_val, x_value, interv_left, interv_right)}/{len(s_val)}"
            )
            print(
                f"Testing {in_fill_between(s_test, n_test, x_value, interv_left, interv_right)}/{len(s_test)}"
            )

        def interval_plot_func(pred, interv_left, interv_right, color, name):
            # Plot predictions and intervals.
            ax.plot(pred, x_value, color=color, zorder=10)
            if np.isfinite(interv_left).all() and np.isfinite(interv_right).all():
                ax.fill_betweenx(
                    x_value,
                    interv_left,
                    interv_right,
                    alpha=0.4,
                    color=color,
                    edgecolor=None,
                    label=name,
                    zorder=0,
                )
                print(name)
                report(interv_left, interv_right)
            else:
                warnings.warn(
                    f"Invalid value encountered when calculating intervals. It is probably because only one"
                    f"unique stress value exists in the training set."
                )

        def psn_plot_func(pred, color, name):
            # Plot psn
            ax.plot(pred, x_value, "--", color=color, label=name)

        ax, given_ax = self._plot_action_init_ax(ax, figure_kwargs_)

        # Plot datasets.
        if train_val_test_color is None:
            train_val_test_color = global_palette[:3]
        if plot_scatter:
            scatter_plot_func(n_train, s_train, train_val_test_color[0], "Training")
            scatter_plot_func(n_val, s_val, train_val_test_color[1], "Validation")
            scatter_plot_func(n_test, s_test, train_val_test_color[2], "Testing")

        # Plot predictions and intervals.
        if len(m_train_indices) > 3 and plot_pof_area:
            _, ci_left, ci_right, psn_pred = get_interval_psn(
                s_train.values,
                n_train.values,
                x_value,
                n_pred_vals=mean_pred,
            )

            interval_plot_func(
                mean_pred, ci_left, ci_right, global_palette[1], f"{model_name} CI"
            )
            psn_plot_func(
                psn_pred, color=global_palette[1], name=f"{model_name} 5\% PoF"
            )
        else:
            if (
                not (np.isnan(ci_left).any() or np.isnan(ci_right).any())
            ) and plot_pof_area:
                interval_plot_func(
                    mean_pred,
                    ci_left,
                    ci_right,
                    global_palette[1],
                    f"Bootstrap {model_name} CI {CI*100:.1f}\%",
                )
            else:
                ax.plot(
                    mean_pred,
                    x_value,
                    **update_defaults_by_kwargs(
                        dict(color=global_palette[1], zorder=10), plot_kwargs
                    ),
                )

        # Get predictions, intervals and psn for lin-log and log-log SN.
        # lin_pred, lin_ci_left, lin_ci_right, lin_psn_pred = get_interval_psn(
        #     s_train.values, n_train.values, x_value
        # )
        # log_pred, log_ci_left, log_ci_right, log_psn_pred = get_interval_psn(
        #     np.log10(s_train.values), n_train.values, np.log10(x_value)
        # )

        # Plot predictions, intervals and psn.
        # interval_plot_func(lin_pred, lin_ci_left, lin_ci_right, global_palette[0], f"Lin-log CI")

        # interval_plot_func(log_pred, log_ci_left, log_ci_right, global_palette[2], f"Log-log CI")

        # psn_plot_func(lin_psn_pred, color=global_palette[0], name=f"Lin-log 5\% PoF")
        # psn_plot_func(log_psn_pred, color=global_palette[2], name=f"Log-log 5\% PoF")

        ax.legend(**legend_kwargs_)
        if log_stress:
            ax.set_yscale("log")
        # ax.set_xlim([0, 10])
        ax.set_title(f"{m_code}")

        path = os.path.join(self.project_root, f"SN_curves_{program}_{model_name}")
        returned = self._plot_action_after_plot(
            fig_name=os.path.join(
                path,
                m_code.replace("/", "_") + ".pdf",
            ),
            disable=given_ax,
            ax_or_fig=ax,
            xlabel=n_col,
            ylabel=s_col,
            tight_layout=False,
            save_show_close=save_show_close,
            savefig_kwargs=savefig_kwargs,
        )
        return returned if inspects is None else (returned, inspects)

    @staticmethod
    def _sn_interval(
        method: str,
        y: np.ndarray,
        y_pred: np.ndarray,
        x: np.ndarray,
        xvals: np.ndarray,
        CI: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate SN confidence intervals based on statistical or ASTM method.

        Parameters
        ----------
        method
            "statistical" or "astm".

            "statistical": Schneider, C. R. A., and S. J. Maddox. "Best practice guide on statistical analysis of
            fatigue data." Weld Inst Stat Rep (2003).

            "astm": According to ASTM E739-10(2015). x can be stress, log(stress), strain, log(strain), etc. It is
            valid when y and x follow the linear assumption. Barbosa, Joelton Fonseca, et al. "Probabilistic SN fields
            based on statistical distributions applied to metallic and composite materials: State of the art." Advances
            in Mechanical Engineering 11.8 (2019): 1687814019870395.
        y
            The true value of fatigue life (in log scale).
        y_pred
            The predicted value of fatigue life (in log scale).
        x
            The true value of stress.
        xvals
            The value of stress to be evaluated on.
        CI
            The confidence interval.

        Returns
        -------
        ci
            The widths of left and right confidence bounds.
        """
        n = len(x)
        STEYX = (
            ((y.reshape(1, -1) - y_pred.reshape(1, -1)) ** 2).sum() / (n - 2)
        ) ** 0.5
        DEVSQ = ((x - np.mean(x)) ** 2).sum().reshape(1, -1)
        if method == "statistical":
            # The two-sided prediction limits are symmetrical, so we calculate one-sided limit instead; therefore, in
            # st.t.ppf or st.f.ppf, the first probability argument is (CI+1)/2 instead of CI for one-sided prediction limit.
            # Because, for example for two-sided CI=95%, the lower limit is equivalent to one-sided 97.5% limit.
            tinv = st.t.ppf((CI + 1) / 2, n - 2)
            CL = tinv * STEYX * (1 + 1 / n + (xvals - np.mean(x)) ** 2 / DEVSQ) ** 0.5
            return CL.flatten(), CL.flatten()
        elif method == "astm":
            # The first parameter is CI instead of (CI+1)/2 according to the ASTM standard. We verified have verified
            # this point by reproducing its given example in Section 8.3 using the following code:
            # from src.core.trainer import Trainer
            # import numpy as np
            # import scipy.stats as st
            # from sklearn.linear_model import LinearRegression
            # x = np.array([-1.78622, -1.79344, -2.17070, -2.16622, -2.74715, -2.79588, -2.78252, -3.27252, -3.26761])
            # y = np.array([2.22531, 2.30103, 3., 3.07188, 3.67486, 3.90499, 3.72049, 4.45662, 4.51388])
            # lr = LinearRegression()
            # lr.fit(x.reshape(-1, 1), y.reshape(-1,1))
            # y_pred = lr.predict(x.reshape(-1,1))
            # xvals = np.linspace(np.min(x), np.max(x), 100)
            # cl, cr = Trainer._sn_interval("ASTM", y, y_pred, x, xvals, 0.95)
            # print(xvals[-15]) #-1.9964038383838383
            # print(cl[-15]) #0.15220609531569082, comparable with given 0.15215. Differences might come from the
            # # regression coefficients.
            tinv = st.f.ppf(CI, 2, n - 2)
            CL = (
                np.sqrt(2 * tinv)
                * STEYX
                * (1 / n + (xvals - np.mean(x)) ** 2 / DEVSQ) ** 0.5
            )
            return CL.flatten(), CL.flatten()
        else:
            raise Exception(f"S-N interval type {method} not implemented.")

    @staticmethod
    def _psn(
        method: str,
        y: np.ndarray,
        y_pred: np.ndarray,
        x: np.ndarray,
        xvals: np.ndarray,
        CI: float,
        p: float,
    ) -> np.ndarray:
        """
        Calculate probabilistic SN curves.

        Parameters
        ----------
        method
            "iso" is currently supported. See ISO 12107.
        y
            The true value of fatigue life (in log scale).
        y_pred
            The predicted value of fatigue life (in log scale).
        x
            The true value of stress.
        xvals
            The value of stress to be evaluated on.
        CI
            The confidence interval.
        p
            The probability to failure.

        Returns
        -------
        cl
            The probabilistic SN curve.
        """
        n = len(x)
        STEYX = (
            ((y.reshape(1, -1) - y_pred.reshape(1, -1)) ** 2).sum() / (n - 2)
        ) ** 0.5
        DEVSQ = ((x - np.mean(x)) ** 2).sum().reshape(1, -1)
        if method == "iso":

            def oneside_normal(p, CI, sample_size, n_random=100000, ddof=1):
                # The one-sided tolerance limits of normal distribution in ISO are given in a table. We find that the
                # analytical calculation is difficult to implement (https://statpages.info/tolintvl.html gives an
                # interactive implementation and a .xls file). We use Monte Carlo simulation to get a more precise
                # value. Since the value is calculated once per plot, the cost is affordable.
                # Refs:
                # https://stackoverflow.com/questions/63698305/how-to-calculate-one-sided-tolerance-interval-with-scipy
                # (or https://jekel.me/tolerance_interval_py/oneside/oneside.html)
                from scipy.stats import norm, nct

                p = 1 - p if p < 0.5 else p
                x_tmp = np.random.randn(n_random, sample_size)
                sigma_est = x_tmp.std(axis=1, ddof=ddof)
                zp = norm.ppf(p)
                t = nct.ppf(CI, df=sample_size - ddof, nc=np.sqrt(sample_size) * zp)
                k = t / np.sqrt(sample_size)
                return np.mean(k * sigma_est)

            k = oneside_normal(p=p, CI=CI, sample_size=n - 2)
            CL = k * STEYX * (1 + 1 / n + (xvals - np.mean(x)) ** 2 / DEVSQ) ** 0.5
            return CL.flatten()
        else:
            raise Exception(f"P-S-N type {method} not implemented.")
