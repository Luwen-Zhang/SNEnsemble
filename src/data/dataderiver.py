from tabensemb.utils import *
from tabensemb.data import AbstractDeriver
from tabensemb.data.utils import get_corr_sets
import inspect


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
                "Absolute Stress Amplitude",
                "Absolute Mean Stress",
            ]
        if self.kwargs["relative"]:
            names += [
                "Relative Maximum Stress",
                "Relative Stress Amplitude",
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
        df_tmp["Absolute Stress Amplitude"] = (
            np.abs(df_tmp[max_stress_col] - df_tmp[min_stress_col]) / 2
        )
        df_tmp["Absolute Mean Stress"] = (
            df_tmp[max_stress_col] + df_tmp[min_stress_col]
        ) / 2
        df_tmp["Relative Maximum Stress"] = np.abs(
            df_tmp["Absolute Maximum Stress"] / df_tmp["rt"]
        )
        df_tmp["Relative Stress Amplitude"] = np.abs(
            df_tmp["Absolute Stress Amplitude"] / df_tmp["rt"]
        )
        df_tmp["Relative Mean Stress"] = np.abs(
            df_tmp["Absolute Mean Stress"] / df_tmp["rt"]
        )
        where_invalid = df_tmp.index[
            np.where(df_tmp["Relative Maximum Stress"] > 1.0)[0]
        ]
        df_tmp.loc[where_invalid, "Relative Maximum Stress"] = np.nan
        df_tmp.loc[where_invalid, "Relative Stress Amplitude"] = np.nan
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


class LayUpSequenceDeriver(AbstractDeriver):
    """
    Derive a padded array according to the laminate sequence code. For instance, if the length of the longest sequence
    is 7, then "0/45/90/45/0" is derived as [0, 45, 90, 45, 0, 100, 100], where the last two "100" are the padding
    value which should be ignored. Missing value is filled as 0 (degree).Required arguments are:
    sequence_column: str
        The column of laminate sequence codes (for instance, "0/45/90/45/0").
    """

    def _required_cols(self):
        return ["sequence_column"]

    def _required_kwargs(self):
        return []

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


class NumLayersDeriver(AbstractDeriver):
    """
    Derive the total number of layers according to the laminate sequence code.
    Missing value will be filled as 1 (layer). Required arguments are:
    sequence_column: str
        The column of laminate sequence codes (for instance, "0/45/90/45/0").
    """

    def _required_cols(self):
        return ["sequence_column"]

    def _required_kwargs(self):
        return []

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


mapping = {}
clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for name, cls in clsmembers:
    if issubclass(cls, AbstractDeriver):
        mapping[name] = cls

tabensemb.data.dataderiver.deriver_mapping.update(mapping)
