import numpy as np
import sys, inspect


class AbstractDeriver:
    def __init__(self):
        pass

    def derive(
        self,
        df,
        trainer,
        derived_name,
        col_names,
        stacked,
        intermediate=False,
        **kwargs,
    ):
        raise NotImplementedError

    @staticmethod
    def _generate_col_names(derived_name, length, col_names):
        if col_names is None or len(col_names) != length:
            names = (
                [f"{derived_name}-{idx}" for idx in range(length)]
                if length > 1
                else [derived_name]
            )
        else:
            names = col_names
        return names

    def _check_arg(self, arg, name):
        if arg is None:
            raise Exception(
                f"Derivation: {name} should be specified for deriver {self.__class__.__name__}"
            )

    def _check_exist(self, df, arg, name):
        if arg not in df.columns:
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

    def derive(
        self,
        df,
        trainer,
        derived_name=None,
        col_names=None,
        stacked=True,
        intermediate=False,
        sequence_column=None,
        **kwargs,
    ):
        self._check_arg(derived_name, "derived_name")
        self._check_arg(sequence_column, "sequence_column")
        self._check_exist(df, sequence_column, "sequence_column")

        sequence = [
            [int(y) if y != "nan" else np.nan for y in str(x).split("/")]
            for x in df["Sequence"].values
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

        names = self._generate_col_names(derived_name, deg_layers.shape[1], col_names)

        self._check_values(deg_layers)
        return deg_layers, derived_name, names, intermediate


class RelativeDeriver(AbstractDeriver):
    def __init__(self):
        super(RelativeDeriver, self).__init__()

    def derive(
        self,
        df,
        trainer,
        derived_name=None,
        col_names=None,
        stacked=True,
        intermediate=False,
        absolute_col=None,
        relative2_col=None,
        **kwargs,
    ):
        self._check_arg(derived_name, "derived_name")
        self._check_arg(absolute_col, "absolute_col")
        self._check_arg(relative2_col, "relative2_col")
        self._check_exist(df, absolute_col, "absolute_col")
        self._check_exist(df, relative2_col, "relative2_col")

        relative = df[absolute_col] / df[relative2_col]
        relative = relative.values.reshape(-1, 1)

        names = self._generate_col_names(derived_name, 1, col_names)
        self._check_values(relative)
        return relative, derived_name, names, intermediate


class MinStressDeriver(AbstractDeriver):
    def __init__(self):
        super(MinStressDeriver, self).__init__()

    def derive(
        self,
        df,
        trainer,
        derived_name=None,
        col_names=None,
        stacked=True,
        intermediate=False,
        max_stress_col=None,
        r_value_col=None,
        **kwargs,
    ):
        self._check_arg(derived_name, "derived_name")
        self._check_arg(max_stress_col, "max_stress_col")
        self._check_arg(r_value_col, "r_value_col")
        self._check_exist(df, max_stress_col, "max_stress_col")
        self._check_exist(df, r_value_col, "r_value_col")

        value = (df[max_stress_col] * df[r_value_col]).values.reshape(-1, 1)

        names = self._generate_col_names(derived_name, 1, col_names)
        self._check_values(value)
        return value, derived_name, names, intermediate


class WalkerStressDeriver(AbstractDeriver):
    def __init__(self):
        super(WalkerStressDeriver, self).__init__()

    def derive(
        self,
        df,
        trainer,
        derived_name=None,
        col_names=None,
        stacked=True,
        intermediate=False,
        max_stress_col=None,
        r_value_col=None,
        power_index=None,
        **kwargs,
    ):
        self._check_arg(derived_name, "derived_name")
        self._check_arg(max_stress_col, "max_stress_col")
        self._check_arg(r_value_col, "r_value_col")
        self._check_arg(power_index, "power_index")
        self._check_exist(df, max_stress_col, "max_stress_col")
        self._check_exist(df, r_value_col, "r_value_col")

        value = (
            df[max_stress_col] * ((1 - df[r_value_col]) / 2) ** power_index
        ).values.reshape(-1, 1)

        names = self._generate_col_names(derived_name, 1, col_names)
        self._check_values(value)
        return value, derived_name, names, intermediate


class SuppStressDeriver(AbstractDeriver):
    def __init__(self):
        super(SuppStressDeriver, self).__init__()

    def derive(
        self,
        df,
        trainer,
        derived_name=None,
        col_names=None,
        stacked=True,
        intermediate=False,
        max_stress_col=None,
        min_stress_col=None,
        ucs_col=None,
        uts_col=None,
        relative=False,
        **kwargs,
    ):
        self._check_arg(derived_name, "derived_name")
        self._check_arg(max_stress_col, "max_stress_col")
        self._check_arg(min_stress_col, "min_stress_col")
        self._check_arg(ucs_col, "ucs_col")
        self._check_arg(uts_col, "uts_col")
        self._check_exist(df, max_stress_col, "max_stress_col")
        self._check_exist(df, min_stress_col, "min_stress_col")
        self._check_exist(df, ucs_col, "ucs_col")
        self._check_exist(df, uts_col, "uts_col")

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

        names = (
            [
                "Absolute Maximum Stress",
                "Absolute Peak-to-peak Stress",
                "Absolute Mean Stress",
                "Relative Maximum Stress",
                "Relative Peak-to-peak Stress",
                "Relative Mean Stress",
            ]
            if relative
            else [
                "Absolute Maximum Stress",
                "Absolute Peak-to-peak Stress",
                "Absolute Mean Stress",
            ]
        )
        stresses = df_tmp[names].values
        self._check_values(stresses)
        return stresses, derived_name, names, intermediate


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
