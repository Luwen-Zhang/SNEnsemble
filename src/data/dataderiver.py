from src.utils import *
from src.data import AbstractDeriver
from copy import deepcopy as cp
import itertools
import inspect
from scipy.interpolate import CubicSpline
from typing import Type


class DegLayerDeriver(AbstractDeriver):
    """
    Derive the number of layers in 0/45/90/other degree according to the laminate sequence code. Required arguments are:

    sequence_column: str
        The column of laminate sequence codes (for instance, "0/45/90/45/0").
    """

    def __init__(self):
        super(DegLayerDeriver, self).__init__()

    def _required_cols(self, **kwargs):
        return ["sequence_column"]

    def _required_params(self, **kwargs):
        return []

    def _defaults(self):
        return dict(stacked=True, intermediate=False)

    def _derive(
        self,
        df,
        trainer,
        **kwargs,
    ):
        sequence_column = kwargs["sequence_column"]

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
        return dict(stacked=False, intermediate=False)

    def _derive(
        self,
        df,
        trainer,
        **kwargs,
    ):
        sequence_column = kwargs["sequence_column"]

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
        return dict(stacked=False, intermediate=False)

    def _derive(
        self,
        df,
        trainer,
        **kwargs,
    ):
        sequence_column = kwargs["sequence_column"]
        pad_value = 100

        sequence = [
            [int(y) if y != "nan" else 0 for y in str(x).split("/")]
            for x in df[sequence_column].values
        ]

        longest_dim = max([len(x) for x in sequence])

        padded_sequence = [x + [pad_value] * (longest_dim - len(x)) for x in sequence]
        seq = np.array(padded_sequence, dtype=int)

        return seq


class RelativeDeriver(AbstractDeriver):
    """
    Dividing a feature by another to derive a new feature. Required arguments are:

    absolute_col: str
        The feature that needs to be divided.
    relative2_col: str
        The feature that acts as the denominator.
    """

    def __init__(self):
        super(RelativeDeriver, self).__init__()

    def _required_cols(self, **kwargs):
        return ["absolute_col", "relative2_col"]

    def _required_params(self, **kwargs):
        return []

    def _defaults(self):
        return dict(stacked=True, intermediate=False)

    def _derive(
        self,
        df,
        trainer,
        **kwargs,
    ):
        absolute_col = kwargs["absolute_col"]
        relative2_col = kwargs["relative2_col"]

        relative = df[absolute_col] / df[relative2_col]
        relative = relative.values.reshape(-1, 1)

        return relative


class MinStressDeriver(AbstractDeriver):
    """
    Calculate minimum stress using maximum stress and R-value. Required arguments are:

    max_stress_col: str
        The name of maximum stress
    r_value_col: str
        The name of R-value
    """

    def __init__(self):
        super(MinStressDeriver, self).__init__()

    def _required_cols(self, **kwargs):
        return ["max_stress_col", "r_value_col"]

    def _required_params(self, **kwargs):
        return []

    def _defaults(self):
        return dict(stacked=True, intermediate=False)

    def _derive(
        self,
        df,
        trainer,
        **kwargs,
    ):
        max_stress_col = kwargs["max_stress_col"]
        r_value_col = kwargs["r_value_col"]
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

    def __init__(self):
        super(WalkerStressDeriver, self).__init__()

    def _required_cols(self, **kwargs):
        return ["max_stress_col", "r_value_col"]

    def _required_params(self, **kwargs):
        return ["power_index"]

    def _defaults(self):
        return dict(stacked=True, intermediate=False)

    def _derive(
        self,
        df,
        trainer,
        **kwargs,
    ):
        max_stress_col = kwargs["max_stress_col"]
        r_value_col = kwargs["r_value_col"]
        power_index = kwargs["power_index"]
        value = (
            df[max_stress_col] * ((1 - df[r_value_col]) / 2) ** power_index
        ).values.reshape(-1, 1)

        return value


class DriveCoeffDeriver(AbstractDeriver):
    def __init__(self):
        super(DriveCoeffDeriver, self).__init__()

    def _required_cols(self, **kwargs):
        return []

    def _required_params(self, **kwargs):
        return []

    def _defaults(self):
        return dict(stacked=False, intermediate=False)

    def _derive(
        self,
        df,
        trainer,
        **kwargs,
    ):
        data = df.copy()

        mlp_trainer = cp(trainer)
        mlp_trainer.dataderivers = [
            x for x in mlp_trainer.dataderivers if not isinstance(x[0], self.__class__)
        ]
        mlp_trainer.set_data_splitter(name="RandomSplitter")
        mlp_trainer.set_data(
            trainer.df,
            cont_feature_names=trainer.cont_feature_names,
            cat_feature_names=trainer.cat_feature_names,
            label_name=trainer.label_name,
            verbose=False,
            warm_start=True
            if not kwargs["stacked"]
            else False,  # if is stacked, processors are not fit.
            all_training=True,
        )

        from src.model import MLP

        mlp = MLP(mlp_trainer)
        mlp_trainer.modelbases = []
        mlp_trainer.bayes_opt = False
        mlp_trainer.add_modelbases([mlp])
        mlp.train(verbose=False)
        with HiddenPrints():
            x_values_list, mean_pdp_list, _, _ = mlp_trainer.cal_partial_dependence(
                model=mlp,
                model_name="MLP",
                df=mlp_trainer.df,
                derived_data=mlp_trainer.derived_data,
                n_bootstrap=1,
                refit=False,
                resample=False,
                grid_size=5,
                verbose=False,
                rederive=True,
                percentile=80,
            )
        self.avg_pred = mlp_trainer.label_data.values.mean()

        interpolator = {}
        plt.figure(figsize=(10, 10))
        cmap = plt.cm.get_cmap("hsv", len(trainer.cont_feature_names))
        marker = itertools.cycle(("^", "<", ">", "+", "o", "*", "s"))
        ax = plt.subplot(111)

        for i, (x_value, mean_pdp, feature_name) in enumerate(
            zip(x_values_list, mean_pdp_list, trainer.cont_feature_names)
        ):
            # print(
            #     f"{feature_name}, diff_y:{mean_pdp[-1] - mean_pdp[1]:.5f}, diff_x:{x_value[-1] - x_value[0]}"
            # )
            interpolator[feature_name] = (
                CubicSpline(x_value, mean_pdp, bc_type="natural")
                if np.sum(np.abs(x_value)) != 0
                else None
            )
            m = next(marker)
            ax.plot(
                np.linspace(0, 4, 100),
                interpolator[feature_name](np.linspace(x_value[0], x_value[-1], 100))
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
        plt.savefig(os.path.join(trainer.project_root, "trend.pdf"))
        plt.close()
        self.interpolator = interpolator
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


class SuppStressDeriver(AbstractDeriver):
    """
    Calculate absolute values of maximum stress, stress amplitude (Peak-to-peak) and mean stress. Their corresponding
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

    def __init__(self):
        super(SuppStressDeriver, self).__init__()

    def _required_cols(self, **kwargs):
        return ["max_stress_col", "min_stress_col", "ucs_col", "uts_col"]

    def _required_params(self, **kwargs):
        return ["relative"]

    def _defaults(self):
        return dict(stacked=True, intermediate=False, relative=False)

    def _derived_names(self, **kwargs):
        names = (
            [
                "Absolute Maximum Stress",
                "Absolute Peak-to-peak Stress",
                "Absolute Mean Stress",
                "Relative Maximum Stress",
                "Relative Peak-to-peak Stress",
                "Relative Mean Stress",
            ]
            if kwargs["relative"]
            else [
                "Absolute Maximum Stress",
                "Absolute Peak-to-peak Stress",
                "Absolute Mean Stress",
            ]
        )
        return names

    def _derive(
        self,
        df,
        trainer,
        **kwargs,
    ):
        max_stress_col = kwargs["max_stress_col"]
        min_stress_col = kwargs["min_stress_col"]
        ucs_col = kwargs["ucs_col"]
        uts_col = kwargs["uts_col"]

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
        df_tmp["Absolute Peak-to-peak Stress"] = np.abs(
            df_tmp[max_stress_col] - df_tmp[min_stress_col]
        )
        df_tmp["Absolute Mean Stress"] = (
            df_tmp[max_stress_col] + df_tmp[min_stress_col]
        ) / 2
        df_tmp["Relative Maximum Stress"] = np.abs(
            df_tmp["Absolute Maximum Stress"] / df_tmp["rt"]
        )
        df_tmp["Relative Peak-to-peak Stress"] = np.abs(
            df_tmp["Absolute Peak-to-peak Stress"] / df_tmp["rt"]
        )
        df_tmp["Relative Mean Stress"] = np.abs(
            df_tmp["Absolute Mean Stress"] / df_tmp["rt"]
        )
        where_invalid = df_tmp.index[
            np.where(df_tmp["Relative Maximum Stress"] > 1.1)[0]
        ]
        df_tmp.loc[where_invalid, "Relative Maximum Stress"] = np.nan
        df_tmp.loc[where_invalid, "Relative Peak-to-peak Stress"] = np.nan
        df_tmp.loc[where_invalid, "Relative Mean Stress"] = np.nan

        names = self._derived_names(**kwargs)
        stresses = df_tmp[names].values
        return stresses


class SampleWeightDeriver(AbstractDeriver):
    """
    Derive weight for each sample in the dataset.
    """

    def __init__(self):
        super(SampleWeightDeriver, self).__init__()
        self.percentile_dict = {}

    def _required_cols(self, **kwargs):
        return []

    def _required_params(self, **kwargs):
        return []

    def _defaults(self):
        return dict(stacked=False, intermediate=False)

    def _derive(
        self,
        df,
        trainer,
        **kwargs,
    ):
        train_idx = trainer.train_indices
        cont_feature_names = trainer.cont_feature_names
        cat_feature_names = trainer.cat_feature_names
        weight = np.ones((len(df), 1))
        for feature in cont_feature_names:
            # We can only calculate distributions based on known data, i.e. the training set.
            if trainer.training:
                Q1 = np.percentile(
                    df.loc[train_idx, feature].dropna(axis=0), 25, method="midpoint"
                )
                Q3 = np.percentile(
                    df.loc[train_idx, feature].dropna(axis=0), 75, method="midpoint"
                )
                self.percentile_dict[feature] = (Q1, Q3)
            else:
                Q1, Q3 = self.percentile_dict[feature]
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            upper = df.index[np.where(df[feature] >= (Q3 + 1.5 * IQR))[0]]
            lower = df.index[np.where(df[feature] <= (Q1 - 1.5 * IQR))[0]]
            idx = np.union1d(upper, lower)
            if len(idx) == 0:
                continue
            p_outlier = len(idx) / len(df)
            feature_weight = -np.log10(p_outlier)
            weight[idx] *= 1.0 + 0.1 * feature_weight

        for feature in cat_feature_names:
            cnts = df[feature].value_counts()
            unique_values = np.array(cnts.index)
            p_unique_values = cnts.values / len(df)
            feature_weight = np.abs(
                np.log10(p_unique_values) - np.log10(max(p_unique_values))
            )
            for value, w in zip(unique_values, feature_weight):
                where_value = df.index[np.where(df[feature] == value)[0]]
                weight[where_value] *= 1.0 + 0.1 * w

        weight = weight / np.sum(weight) * len(df)
        return weight


deriver_mapping = {}
clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for name, cls in clsmembers:
    if issubclass(cls, AbstractDeriver):
        deriver_mapping[name] = cls


def get_data_deriver(name: str) -> Type[AbstractDeriver]:
    if name not in deriver_mapping.keys():
        raise Exception(f"Data deriver {name} not implemented.")
    elif not issubclass(deriver_mapping[name], AbstractDeriver):
        raise Exception(f"{name} is not the subclass of AbstractDeriver.")
    else:
        return deriver_mapping[name]
