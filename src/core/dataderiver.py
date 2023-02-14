import numpy as np
import sys, inspect
from ..utils.utils import *
from copy import deepcopy as cp


class AbstractDeriver:
    def __init__(self):
        pass

    def derive(
        self,
        df,
        trainer,
        derived_name,
        **kwargs,
    ):
        kwargs = self.make_defaults(**kwargs)
        for arg_name in self._required_cols(**kwargs):
            self._check_arg(arg_name, **kwargs)
            self._check_exist(df, arg_name, **kwargs)
        for arg_name in self._required_params(**kwargs) + ["stacked", "intermediate"]:
            self._check_arg(arg_name, **kwargs)
        values = self._derive(df, trainer, **kwargs)
        self._check_values(values)
        names = (
            self._generate_col_names(
                derived_name, self._derived_size(**kwargs), **kwargs
            )
            if "col_names" not in kwargs
            else kwargs["col_names"]
        )
        return values, derived_name, names

    def make_defaults(self, **kwargs):
        for key, value in self._defaults().items():
            if key not in kwargs.keys():
                kwargs[key] = value
        return kwargs

    def _derive(
        self,
        df,
        trainer,
        **kwargs,
    ):
        raise NotImplementedError

    def _derived_size(self, **kwargs):
        raise NotImplementedError

    def _defaults(self):
        return {}

    def _derived_names(self, **kwargs):
        raise NotImplementedError

    def _generate_col_names(self, derived_name, length, **kwargs):
        try:
            names = self._derived_names(**kwargs)
        except:
            names = (
                [f"{derived_name}-{idx}" for idx in range(length)]
                if length > 1
                else [derived_name]
            )
        return names

    def _required_cols(self, **kwargs):
        raise NotImplementedError

    def _required_params(self, **kwargs):
        raise NotImplementedError

    def _check_arg(self, name, **kwargs):
        if name not in kwargs.keys():
            raise Exception(
                f"Derivation: {name} should be specified for deriver {self.__class__.__name__}"
            )

    def _check_exist(self, df, name, **kwargs):
        if kwargs[name] not in df.columns:
            raise Exception(
                f"Derivation: {name} is not a valid column in df for deriver {self.__class__.__name__}."
            )

    def _check_values(self, values):
        if len(values.shape) == 1:
            raise Exception(
                f"Derivation: {name} returns a one dimensional numpy.ndarray. Use reshape(-1, 1) to transform into 2D."
            )


class DegLayerDeriver(AbstractDeriver):
    def __init__(self):
        super(DegLayerDeriver, self).__init__()

    def _required_cols(self, **kwargs):
        return ["sequence_column"]

    def _required_params(self, **kwargs):
        return []

    def _derived_size(self, **kwargs):
        return 4

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
            (len(sequence), 4), dtype=np.int
        )  # for 0-deg, pm45-deg, 90-deg, and other directions respectively

        for idx, seq in enumerate(sequence):
            deg_layers[idx, 0] = seq.count(0)
            deg_layers[idx, 1] = seq.count(45) + seq.count(-45)
            deg_layers[idx, 2] = seq.count(90)
            deg_layers[idx, 3] = (
                len(seq) - seq.count(np.nan) - np.sum(deg_layers[idx, :3])
            )

        return deg_layers


class RelativeDeriver(AbstractDeriver):
    def __init__(self):
        super(RelativeDeriver, self).__init__()

    def _required_cols(self, **kwargs):
        return ["absolute_col", "relative2_col"]

    def _required_params(self, **kwargs):
        return []

    def _derived_size(self, **kwargs):
        return 1

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
    def __init__(self):
        super(MinStressDeriver, self).__init__()

    def _required_cols(self, **kwargs):
        return ["max_stress_col", "r_value_col"]

    def _required_params(self, **kwargs):
        return []

    def _derived_size(self, **kwargs):
        return 1

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
    def __init__(self):
        super(WalkerStressDeriver, self).__init__()

    def _required_cols(self, **kwargs):
        return ["max_stress_col", "r_value_col"]

    def _required_params(self, **kwargs):
        return ["power_index"]

    def _derived_size(self, **kwargs):
        return 1

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

    def _derived_size(self, **kwargs):
        return 1

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
            feature_names=trainer.feature_names,
            label_name=trainer.label_name,
            verbose=False,
            warm_start=True
            if not kwargs["stacked"]
            else False,  # if is stacked, processors are not fit.
            all_training=True,
        )

        from src.core.model import MLP

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

        from scipy.interpolate import CubicSpline
        import itertools

        interpolator = {}
        plt.figure(figsize=(10, 10))
        cmap = plt.cm.get_cmap("hsv", len(trainer.feature_names))
        marker = itertools.cycle(("^", "<", ">", "+", "o", "*", "s"))
        ax = plt.subplot(111)

        for i, (x_value, mean_pdp, feature_name) in enumerate(
            zip(x_values_list, mean_pdp_list, trainer.feature_names)
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

        _, drive_coeff = self._cal_drive_coeff(df)
        value = drive_coeff.reshape(-1, 1)

        return value

    def _cal_drive_coeff(self, df):
        drive_coeff_feature = {}
        drive_coeff = np.ones((len(df),))
        for feature_name, interpolator in self.interpolator.items():
            drive_coeff_feature[feature_name] = (
                (interpolator(df[feature_name].values).flatten() / self.avg_pred)
                if interpolator is not None
                else np.repeat(1, len(df))
            )

            drive_coeff *= drive_coeff_feature[feature_name]
        return drive_coeff_feature, drive_coeff


class SuppStressDeriver(AbstractDeriver):
    def __init__(self):
        super(SuppStressDeriver, self).__init__()

    def _required_cols(self, **kwargs):
        return ["max_stress_col", "min_stress_col", "ucs_col", "uts_col"]

    def _required_params(self, **kwargs):
        return ["relative"]

    def _defaults(self):
        return dict(stacked=True, intermediate=False, relative=False)

    def _derived_size(self, **kwargs):
        return 6 if kwargs["relative"] else 3

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


deriver_mapping = {}
clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for name, cls in clsmembers:
    if issubclass(cls, AbstractDeriver):
        deriver_mapping[name] = cls


def get_data_deriver(name: str):
    if name not in deriver_mapping.keys():
        raise Exception(f"Data deriver {name} not implemented.")
    elif not issubclass(deriver_mapping[name], AbstractDeriver):
        raise Exception(f"{name} is not the subclass of AbstractDeriver.")
    else:
        return deriver_mapping[name]
