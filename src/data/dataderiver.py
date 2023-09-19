import numpy as np
from tabensemb.utils import *
from tabensemb.data import AbstractDeriver
from tabensemb.data.utils import get_corr_sets
import inspect
from sklearn.inspection import partial_dependence
from sklearn.inspection._partial_dependence import _grid_from_X
from scipy.interpolate import CubicSpline
from numpy.polynomial import Polynomial


class DegLayerDeriver(AbstractDeriver):
    """
    Derive the number of layers in 0/45/90/other degree according to the laminate sequence code. Required arguments are:

    sequence_column: str
        The column of laminate sequence codes (for instance, "0/45/90/45/0").
    """

    def _required_cols(self):
        return ["sequence_column"]

    def _required_kwargs(self):
        return []

    def _defaults(self):
        return dict(stacked=True, intermediate=False, is_continuous=True)

    def _derive(self, df, datamodule):
        sequence_column = self.kwargs["sequence_column"]

        sequence = [
            [int(y) if y != "nan" else np.nan for y in str(x).split("/")]
            for x in df[sequence_column].values
        ]

        deg_layers = np.zeros(
            (len(sequence), 4), dtype=int
        )  # for 0-deg, pm45-deg, 90-deg, and other directions respectively

        for idx, seq in enumerate(sequence):
            deg_layers[idx, 0] = seq.count(0)
            deg_layers[idx, 1] = seq.count(45) + seq.count(-45)
            deg_layers[idx, 2] = seq.count(90)
            deg_layers[idx, 3] = (
                len(seq) - seq.count(np.nan) - np.sum(deg_layers[idx, :3])
            )

        return deg_layers


class NumLayersDeriver(DegLayerDeriver):
    """
    Derive the total number of layers according to the laminate sequence code.
    Missing value will be filled as 1 (layer). Required arguments are:

    sequence_column: str
        The column of laminate sequence codes (for instance, "0/45/90/45/0").
    """

    def _defaults(self):
        return dict(stacked=False, intermediate=False, is_continuous=True)

    def _derive(self, df, datamodule):
        sequence_column = self.kwargs["sequence_column"]

        sequence = [
            [int(y) if y != "nan" else 0 for y in str(x).split("/")]
            for x in df[sequence_column].values
        ]
        n_layers = np.zeros((len(df), 1), dtype=int)
        for idx, x in enumerate(sequence):
            n_layers[idx, 0] = len(x)

        return n_layers


class LayUpSequenceDeriver(DegLayerDeriver):
    """
    Derive a padded array according to the laminate sequence code. For instance, if the length of the longest sequence
    is 7, then "0/45/90/45/0" is derived as [0, 45, 90, 45, 0, 100, 100], where the last two "100" are the padding
    value which should be ignored. Missing value is filled as 0 (degree).Required arguments are:

    sequence_column: str
        The column of laminate sequence codes (for instance, "0/45/90/45/0").
    """

    def _defaults(self):
        return dict(stacked=False, intermediate=False, is_continuous=True)

    def _derive(self, df, datamodule):
        sequence_column = self.kwargs["sequence_column"]
        pad_value = 100

        sequence = [
            [int(y) if y != "nan" else 0 for y in str(x).split("/")]
            for x in df[sequence_column].values
        ]

        longest_dim = max([len(x) for x in sequence])

        padded_sequence = [x + [pad_value] * (longest_dim - len(x)) for x in sequence]
        seq = np.array(padded_sequence, dtype=int)

        return seq


class MinStressDeriver(AbstractDeriver):
    """
    Calculate minimum stress using maximum stress and R-value. Required arguments are:

    max_stress_col: str
        The name of maximum stress
    r_value_col: str
        The name of R-value
    """

    def _required_cols(self):
        return ["max_stress_col", "r_value_col"]

    def _required_kwargs(self):
        return []

    def _defaults(self):
        return dict(stacked=True, intermediate=False, is_continuous=True)

    def _derive(self, df, datamodule):
        max_stress_col = self.kwargs["max_stress_col"]
        r_value_col = self.kwargs["r_value_col"]
        value = (df[max_stress_col] * df[r_value_col]).values.reshape(-1, 1)

        return value


class WalkerStressDeriver(AbstractDeriver):
    """
    Calculate Walker equivalent stress. Required arguments are:

    max_stress_col: str
        The name of maximum stress
    r_value_col: str
        The name of R-value
    power_index: float
        The power index of the walker equivalent stress. If is 0.5, it coincides with SWT equivalent stress.
    """

    def _required_cols(self):
        return ["max_stress_col", "r_value_col"]

    def _required_kwargs(self):
        return ["power_index"]

    def _defaults(self):
        return dict(stacked=True, intermediate=False, is_continuous=True)

    def _derive(self, df, datamodule):
        max_stress_col = self.kwargs["max_stress_col"]
        r_value_col = self.kwargs["r_value_col"]
        power_index = self.kwargs["power_index"]
        value = (
            df[max_stress_col] * ((1 - df[r_value_col]) / 2) ** power_index
        ).values.reshape(-1, 1)

        return value


class SuppStressDeriver(AbstractDeriver):
    """
    Calculate absolute values of maximum stress, stress range and mean stress. Their corresponding
    relative stresses (relative to static modulus) are also calculated. For the maximum stress, the larger one of
    tensile and compressive stresses is selected. Required arguments are:

    max_stress_col: str
        The name of maximum stress
    min_stress_col: str
        The name of minimum stress
    ucs_col: str
        The name of ultimate compressive stress
    uts_col: str
        The name of ultimate tensile stress
    relative: bool
        Whether to calculate relative stresses.
    """

    def _required_cols(self):
        return ["max_stress_col", "min_stress_col", "ucs_col", "uts_col"]

    def _required_kwargs(self):
        return ["relative", "absolute"]

    def _defaults(self):
        return dict(
            stacked=True,
            intermediate=False,
            relative=False,
            absolute=True,
            is_continuous=True,
        )

    def _derived_names(self):
        names = []
        if self.kwargs["absolute"]:
            names += [
                "Absolute Maximum Stress",
                "Absolute Stress Range",
                "Absolute Mean Stress",
            ]
        if self.kwargs["relative"]:
            names += [
                "Relative Maximum Stress",
                "Relative Stress Range",
                "Relative Mean Stress",
            ]
        return names

    def _derive(self, df, datamodule):
        max_stress_col = self.kwargs["max_stress_col"]
        min_stress_col = self.kwargs["min_stress_col"]
        ucs_col = self.kwargs["ucs_col"]
        uts_col = self.kwargs["uts_col"]

        df_tmp = df.copy()

        df_tmp["Absolute Maximum Stress"] = np.maximum(
            np.abs(df_tmp[max_stress_col]), np.abs(df_tmp[min_stress_col])
        )
        where_max = df_tmp.index[
            np.where(
                df_tmp["Absolute Maximum Stress"].values
                == np.abs(df_tmp[max_stress_col])
            )[0]
        ]
        df_tmp.loc[where_max, "Absolute Maximum Stress"] *= np.sign(
            df_tmp.loc[where_max, max_stress_col]
        )
        which_min = np.setdiff1d(df_tmp.index, where_max)
        df_tmp.loc[which_min, "Absolute Maximum Stress"] *= np.sign(
            df_tmp.loc[which_min, min_stress_col]
        )
        where_g0 = df_tmp.index[np.where(df_tmp["Absolute Maximum Stress"] > 0)[0]]
        df_tmp["rt"] = np.abs(df_tmp[ucs_col])
        df_tmp.loc[where_g0, "rt"] = np.abs(df_tmp.loc[where_g0, uts_col])
        df_tmp["Absolute Stress Range"] = np.abs(
            df_tmp[max_stress_col] - df_tmp[min_stress_col]
        )
        df_tmp["Absolute Mean Stress"] = (
            df_tmp[max_stress_col] + df_tmp[min_stress_col]
        ) / 2
        df_tmp["Relative Maximum Stress"] = np.abs(
            df_tmp["Absolute Maximum Stress"] / df_tmp["rt"]
        )
        df_tmp["Relative Stress Range"] = np.abs(
            df_tmp["Absolute Stress Range"] / df_tmp["rt"]
        )
        df_tmp["Relative Mean Stress"] = np.abs(
            df_tmp["Absolute Mean Stress"] / df_tmp["rt"]
        )
        where_invalid = df_tmp.index[
            np.where(df_tmp["Relative Maximum Stress"] > 1.1)[0]
        ]
        df_tmp.loc[where_invalid, "Relative Maximum Stress"] = np.nan
        df_tmp.loc[where_invalid, "Relative Stress Range"] = np.nan
        df_tmp.loc[where_invalid, "Relative Mean Stress"] = np.nan

        names = self._derived_names()
        stresses = df_tmp[names].values
        return stresses


class TheoreticalFiftyPofDeriver(AbstractDeriver):
    def _derive(self, df: pd.DataFrame, datamodule) -> np.ndarray:
        from sklearn.preprocessing import MinMaxScaler
        from scipy import stats

        measure_features = datamodule.cont_feature_names
        target = datamodule.label_name
        distrib = self.kwargs["distribution"]
        distrib_estimator = {
            "gaussian": stats.norm,
            "weibull": stats.exponweib,
        }[distrib]
        mat_lay = df["Material_Code"].copy()
        unique_mat_lay = list(set(mat_lay))
        pof50 = df.loc[:, target].values
        if distrib == "weibull":
            import tqdm.auto

            bar = tqdm.auto.tqdm(total=len(unique_mat_lay))
        else:
            bar = None
        for material in unique_mat_lay:
            where_material = np.where(mat_lay == material)[0]
            df_material = df.loc[where_material, measure_features].copy()
            label_material = df.loc[where_material, target].values.flatten()
            scaler = MinMaxScaler()
            values = scaler.fit_transform(df_material)
            dist = np.sqrt(
                np.sum(np.power((values[:, None, :] - values[None, :, :]), 2), axis=-1)
            )
            where_same = np.where(dist < 0.05)
            _, same_sets = get_corr_sets(where_same, list(np.arange(len(df_material))))
            for same_set in same_sets:
                target_set = label_material[same_set]
                unique_vals = len(set(target_set))
                if unique_vals < 2:
                    continue
                warnings.filterwarnings("ignore", "invalid value encountered in add")
                pof50[where_material[same_set]] = distrib_estimator.ppf(
                    0.5, *distrib_estimator.fit(target_set)
                )
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()
        return pof50

    @staticmethod
    def describe_acc(target, pof50, datamodule, distribution):
        from sklearn.metrics import mean_squared_error

        train_mse = mean_squared_error(
            target[datamodule.train_indices], pof50[datamodule.train_indices]
        )
        train_rmse = np.sqrt(train_mse)
        val_mse = mean_squared_error(
            target[datamodule.val_indices], pof50[datamodule.val_indices]
        )
        val_rmse = np.sqrt(val_mse)
        test_mse = mean_squared_error(
            target[datamodule.test_indices], pof50[datamodule.test_indices]
        )
        test_rmse = np.sqrt(test_mse)
        mse = mean_squared_error(target, pof50)
        rmse = np.sqrt(mse)
        print(f"Theoretical accuracy at PoF=50% ({distribution} distribution):")
        print(
            f"\t Training MSE {train_mse:.5f}, RMSE {train_rmse:.5f}\n"
            f"\t Validation MSE {val_mse:.5f}, RMSE {val_rmse:.5f}\n"
            f"\t Testing MSE {test_mse:.5f}, RMSE {test_rmse:.5f}\n"
            f"\t Overall MSE {mse:.5f}, RMSE {rmse:.5f}"
        )

    def _defaults(self):
        return dict(
            stacked=True, intermediate=True, distribution="gaussian", is_continuous=True
        )

    def _derived_names(self):
        return ["TheoreticalFiftyPof"]

    def _required_cols(self):
        return []

    def _required_kwargs(self):
        return []


class TrendDeriver(AbstractDeriver):
    def _required_cols(self):
        return []

    def _required_kwargs(self):
        return []

    def _defaults(self):
        return dict(stacked=False, intermediate=False, plot=False, is_continuous=True)

    def _derive(self, df: pd.DataFrame, datamodule) -> np.ndarray:
        try:
            df = datamodule.categories_transform(df)
        except:
            pass
        if datamodule.training:
            predictor = datamodule.get_base_predictor(categorical=False)
            grid_resolution = 5

            x_value_list = []
            pdp_list = []

            df_train = df.loc[datamodule.train_indices, :]
            feature_data = df_train[datamodule.cont_feature_names]
            label_data = df_train[datamodule.label_name].values.flatten()
            predictor.fit(feature_data, label_data)

            importances = predictor.feature_importances_
            order = np.argsort(importances)[::-1]
            n_trend_features = np.where(importances[order].cumsum() < 0.8)[0][-1] + 1
            trend_features = [datamodule.cont_feature_names[x] for x in order][
                :n_trend_features
            ]

            cont_grids = {}
            for feature in trend_features:
                _, x_value_interp = _grid_from_X(
                    df[[feature]].values, (0.05, 0.95), grid_resolution
                )
                cont_grids[feature] = x_value_interp[0]
            # cat_grids = {}
            # for feature in datamodule.cat_feature_names:
            #     cat_grids[feature] = np.sort(np.unique(df[feature].values))

            for feature in trend_features:
                pdp_result = partial_dependence(
                    predictor,
                    feature_data,
                    [feature],
                    grid_resolution=grid_resolution,
                    kind="average",
                    method="auto",
                )
                model_predictions = pdp_result["average"][0]
                x_value = pdp_result["values"][0]

                pdp_list.append(
                    np.interp(cont_grids[feature], x_value, model_predictions)
                )
                x_value_list.append(cont_grids[feature])
                # predictions = []
                # for val in cat_grids[feature]:
                #     predictions.append(
                #         model_predictions[np.where(x_value == val)[0][0]]
                #     )
                # pdp_list.append(np.array(predictions))
                # x_value_list.append(cat_grids[feature])

            self.avg_pred = np.mean(df[datamodule.label_name].values)

            interpolator = {}
            for i, (x_value, mean_pdp, feature_name) in enumerate(
                zip(x_value_list, pdp_list, trend_features)
            ):
                if np.sum(np.abs(x_value)) != 0:
                    deg1_fit, _ = Polynomial.fit(x_value, mean_pdp, deg=1, full=True)
                    deg1_gof = auto_metric_sklearn(
                        mean_pdp, deg1_fit(x_value), "r2", "regression"
                    )
                    deg2_fit, _ = Polynomial.fit(x_value, mean_pdp, deg=2, full=True)
                    deg2_gof = auto_metric_sklearn(
                        mean_pdp, deg2_fit(x_value), "r2", "regression"
                    )
                    interpolator[feature_name] = (
                        deg1_fit if deg2_gof - deg1_gof < 0.1 else deg2_fit
                    )
                else:
                    interpolator[feature_name] = None
            self.interpolator = interpolator
            if self.kwargs["plot"]:
                plt.figure(figsize=(10, 10))
                cmap = plt.cm.get_cmap("hsv", len(trend_features))
                marker = itertools.cycle(("^", "<", ">", "+", "o", "*", "s"))
                ax = plt.subplot(111)
                for i, (x_value, mean_pdp, feature_name) in enumerate(
                    zip(x_value_list, pdp_list, trend_features)
                ):
                    # if feature_name in datamodule.cat_feature_names:
                    #     continue
                    m = next(marker)
                    ax.plot(
                        np.linspace(0, 4, 100),
                        interpolator[feature_name](
                            np.linspace(x_value[0], x_value[-1], 100)
                        )
                        if interpolator[feature_name] is not None
                        else np.repeat(self.avg_pred, 100),
                        linestyle="-",
                        c=cmap(i),
                        marker=m,
                        label=feature_name,
                        markevery=1000,
                    )
                    ax.scatter(
                        np.arange(len(x_value)),
                        mean_pdp
                        if interpolator[feature_name] is not None
                        else np.repeat(self.avg_pred, 5),
                        c=[cmap(i) for x in x_value],
                        marker=m,
                    )
                ax.set_xticks([0, 1, 2, 3, 4])
                plt.legend(fontsize="small")
                plt.show()
                plt.close()
        drive_coeff = self._cal_drive_coeff(df)
        return drive_coeff

    def _cal_drive_coeff(self, df):
        drive_coeff = np.zeros((len(df), len(self.interpolator)))
        for idx, (feature_name, interpolator) in enumerate(self.interpolator.items()):
            drive_coeff[:, idx] = (
                (interpolator(df[feature_name].values).flatten() / self.avg_pred)
                if interpolator is not None
                else np.repeat(1, len(df))
            )
        return drive_coeff


mapping = {}
clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for name, cls in clsmembers:
    if issubclass(cls, AbstractDeriver):
        mapping[name] = cls

tabensemb.data.dataderiver.deriver_mapping.update(mapping)
