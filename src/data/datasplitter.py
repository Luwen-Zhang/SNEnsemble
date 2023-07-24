import numpy as np
from tabensemb.utils import *
from tabensemb.data import AbstractSplitter
import inspect
from sklearn.model_selection import train_test_split
from typing import Type, List, Tuple


def mat_lay_index(chosen_mat_lay, mat_lay):
    index = []
    for material in chosen_mat_lay:
        where_material = np.where(mat_lay == material)[0]
        index += list(where_material)
    return np.array(index)


class MaterialSplitter(AbstractSplitter):
    """
    Split the dataset by the material code to simulate the scenario of designing new materials.
    Training/validation/testing datasets will contain entirely different materials.
    """

    def _split(self, df, cont_feature_names, cat_feature_names, label_name):
        self._check_exist(df, "Material_Code", "Material_Code")
        mat_lay = np.array([str(x) for x in df["Material_Code"].copy()])
        mat_lay_set = list(sorted(set(mat_lay)))

        train_mat_lay, val_mat_lay, test_mat_lay = self.split_method(
            mat_lay_set, self.train_val_test
        )

        return (
            mat_lay_index(train_mat_lay, mat_lay),
            mat_lay_index(val_mat_lay, mat_lay),
            mat_lay_index(test_mat_lay, mat_lay),
        )

    @classmethod
    def split_method(cls, mat_lay_set, train_val_test):
        train_mat_lay, test_mat_lay = train_test_split(
            mat_lay_set, test_size=train_val_test[2], shuffle=True
        )
        train_mat_lay, val_mat_lay = train_test_split(
            train_mat_lay,
            test_size=train_val_test[1] / np.sum(train_val_test[0:2]),
            shuffle=True,
        )
        return train_mat_lay, val_mat_lay, test_mat_lay

    @property
    def support_cv(self):
        return True

    def _next_cv(
        self,
        df: pd.DataFrame,
        cont_feature_names: List[str],
        cat_feature_names: List[str],
        label_name: List[str],
        cv: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        convert_to_str = lambda x: np.array([str(i) for i in x])
        self._check_exist(df, "Material_Code", "Material_Code")
        mat_lay = convert_to_str(df["Material_Code"].copy())
        mat_lay_set = list(sorted(set(mat_lay)))

        train_mat_lay, val_mat_lay, test_mat_lay = self._sklearn_k_fold(mat_lay_set, cv)

        train_mat_lay = convert_to_str(train_mat_lay)
        val_mat_lay = convert_to_str(val_mat_lay)
        test_mat_lay = convert_to_str(test_mat_lay)
        return (
            mat_lay_index(train_mat_lay, mat_lay),
            mat_lay_index(val_mat_lay, mat_lay),
            mat_lay_index(test_mat_lay, mat_lay),
        )


class MaterialCycleSplitter(AbstractSplitter):
    """
    Split the dataset by the material code and the number of cycles to simulate the scenario of designing new materials
    using limited data from accelerated fatigue tests. Training/validation/testing datasets will contain entirely
    different materials, and validation/testing sets contain data that generally have larger number of cycles than the
    training set (even much larger in the testing set).
    """

    def _split(self, df, cont_feature_names, cat_feature_names, label_name):
        self.train_ratio = np.sqrt(self.train_val_test[0])
        self.val_ratio = (
            (1 - self.train_ratio)
            * self.train_val_test[1]
            / np.sum(self.train_val_test[1:])
        )
        self.test_ratio = (
            (1 - self.train_ratio)
            * self.train_val_test[2]
            / np.sum(self.train_val_test[1:])
        )
        self._check_exist(df, "Material_Code", "Material_Code")
        mat_lay = np.array([str(x) for x in df["Material_Code"].copy()])
        mat_lay_set = list(sorted(set(mat_lay)))
        if "log" not in label_name and "Cycles to Failure" not in label_name:
            raise Exception(
                f"{self.__class__.__name__} requires log10(Cycles to Failure) being the target, but only {label_name} exist."
            )
        cycle = df[label_name].values.flatten()
        freq = df["Frequency"].values.flatten() if "Frequency" in df.columns else None
        r_value = df["R-value"].values.flatten() if "R-value" in df.columns else None

        train_mat_lay, val_mat_lay, test_mat_lay = MaterialSplitter.split_method(
            mat_lay_set, [self.train_ratio, self.val_ratio, self.test_ratio]
        )

        mat_val_indices = mat_lay_index(val_mat_lay, mat_lay)
        mat_test_indices = mat_lay_index(test_mat_lay, mat_lay)

        train_indices, val_indices, test_indices = CycleSplitter.split_method(
            train_mat_lay,
            mat_lay,
            cycle,
            [self.train_ratio, self.val_ratio, self.test_ratio],
            freq=freq,
            r_value=r_value,
        )

        val_indices = np.append(val_indices, mat_val_indices)
        test_indices = np.append(test_indices, mat_test_indices)

        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)

        return np.array(train_indices), np.array(val_indices), np.array(test_indices)


class CycleSplitter(AbstractSplitter):
    """
    Split the dataset by the material code and the number of cycles to simulate the scenario of prediction using
    limited data from accelerated fatigue tests. Validation/testing sets contain data that have larger number of cycles
    than the training set (even much larger in the testing set). If given at least one of "Frequency" and "R-value",
    the splitting is performed for each material and for each combination of frequency-R (instead of only for each
    material).

    Steps of the splitting are:
    1. Find data points for each material and for each combination of frequency-R.
    2. In these data points, find 60% points with lower Nf for the training set, and the rest of them are for validation
       and testing.
    3. In the 40% part, validation and testing sets are randomly split by train_test_split. If only one point is
       available, it will be randomly decided as a validation or testing point.
    4. To simulation the real scenario and to prevent over-fitting on the validation set, the training and validation
       sets are completely shuffled.
    """

    def _split(self, df, cont_feature_names, cat_feature_names, label_name):
        self._check_exist(df, "Material_Code", "Material_Code")
        mat_lay = np.array([str(x) for x in df["Material_Code"].copy()])
        mat_lay_set = list(sorted(set(mat_lay)))
        if "log" not in label_name and "Cycles to Failure" not in label_name:
            raise Exception(
                f"{self.__class__.__name__} requires log10(Cycles to Failure) being the target, but only {label_name} exist."
            )
        cycle = df[label_name].values.flatten()
        freq = df["Frequency"].values.flatten() if "Frequency" in df.columns else None
        r_value = df["R-value"].values.flatten() if "R-value" in df.columns else None

        train_indices, val_indices, test_indices = self.split_method(
            mat_lay_set, mat_lay, cycle, self.train_val_test, freq=freq, r_value=r_value
        )

        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)

        return np.array(train_indices), np.array(val_indices), np.array(test_indices)

    @classmethod
    def split_method(
        cls, mat_lay_set, mat_lay, cycle, train_val_test, freq=None, r_value=None
    ):
        train_indices = []
        val_indices = []
        test_indices = []

        if freq is not None and r_value is not None:
            freq_r = np.array([f"{f}_{r}" for f, r in zip(freq, r_value)])
        elif freq is None and r_value is None:
            freq_r = None
        else:
            freq_r = freq if freq is not None else r_value

        for material in mat_lay_set:
            where_material = np.where(mat_lay == material)[0]
            material_cycle = cycle[where_material]
            if freq_r is not None:
                material_freq_r = freq_r[where_material]
                m_train_indices = []
                m_test_indices = []
                m_val_indices = []
                for fr in np.unique(material_freq_r):
                    if not isinstance(fr, str) and np.isnan(fr):
                        where_fr = np.where(np.isnan(material_freq_r))[0]
                        m_train_indices += list(where_material[where_fr])
                        continue
                    where_fr = np.where(material_freq_r == fr)[0]
                    fr_cycle = material_cycle[where_fr]
                    (
                        fr_train_indices,
                        fr_val_indices,
                        fr_test_indices,
                    ) = cls._split_one_fr(where_fr, fr_cycle, train_val_test)
                    m_train_indices += list(where_material[fr_train_indices])
                    m_test_indices += list(where_material[fr_test_indices])
                    m_val_indices += list(where_material[fr_val_indices])
            else:
                m_train_indices = where_material[
                    material_cycle
                    <= np.percentile(material_cycle, train_val_test[0] * 100)
                ]
                m_test_indices = where_material[
                    material_cycle
                    > np.percentile(material_cycle, np.sum(train_val_test[0:2]) * 100)
                ]
                m_val_indices = np.setdiff1d(
                    where_material, np.append(m_train_indices, m_test_indices)
                )

            train_indices += list(m_train_indices)
            val_indices += list(m_val_indices)
            test_indices += list(m_test_indices)

        train_val_indices = np.concatenate([train_indices, val_indices])
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=train_val_test[1] / np.sum(train_val_test[:2]),
            shuffle=True,
        )
        return np.array(train_indices), np.array(val_indices), np.array(test_indices)

    @classmethod
    def _split_one_fr(cls, where_fr, fr_cycle, train_val_test):
        fr_train_indices = np.where(
            fr_cycle <= np.percentile(fr_cycle, train_val_test[0] * 100)
        )[0]
        fr_test_val_indices = np.setdiff1d(np.arange(len(fr_cycle)), fr_train_indices)
        if len(fr_test_val_indices) >= np.sum(train_val_test[1:]) // train_val_test[-1]:
            fr_val_indices, fr_test_indices = train_test_split(
                fr_test_val_indices,
                test_size=train_val_test[-1] / np.sum(train_val_test[1:]),
                shuffle=True,
            )
        elif len(fr_test_val_indices) > 1:
            fr_val_indices, fr_test_indices = train_test_split(
                fr_test_val_indices,
                test_size=1 / len(fr_test_val_indices),
                shuffle=True,
            )
        else:
            if np.random.randint(2) == 0:
                fr_val_indices = fr_test_val_indices
                fr_test_indices = np.array([], dtype=int)
            else:
                fr_test_indices = fr_test_val_indices
                fr_val_indices = np.array([], dtype=int)
        return (
            where_fr[fr_train_indices],
            where_fr[fr_val_indices],
            where_fr[fr_test_indices],
        )


class StrictCycleSplitter(CycleSplitter):
    """
    Compared to CycleSplitter, the third step is different:

    3. In the 40% part, data points with lower Nf are selected as the validation set. If only one point isavailable,
    it will be randomly decided as a validation or testing point. If the number of points is an odd number, the
    validation set would have one more point than the testing set.
    """

    @classmethod
    def _split_one_fr(cls, where_fr, fr_cycle, train_val_test):
        fr_train_indices = np.where(
            fr_cycle <= np.percentile(fr_cycle, train_val_test[0] * 100)
        )[0]
        fr_test_val_indices = np.setdiff1d(np.arange(len(fr_cycle)), fr_train_indices)
        if len(fr_test_val_indices) > 1:
            fr_test_indices = fr_test_val_indices[
                fr_cycle[fr_test_val_indices]
                > np.percentile(
                    fr_cycle[fr_test_val_indices],
                    train_val_test[1] / np.sum(train_val_test[1:]) * 100,
                )
            ]
            fr_val_indices = np.setdiff1d(fr_test_val_indices, fr_test_indices)
        else:
            if np.random.randint(2) == 0:
                fr_val_indices = fr_test_val_indices
                fr_test_indices = np.array([], dtype=int)
            else:
                fr_test_indices = fr_test_val_indices
                fr_val_indices = np.array([], dtype=int)
        return (
            where_fr[fr_train_indices],
            where_fr[fr_val_indices],
            where_fr[fr_test_indices],
        )


class StressCycleSplitter(StrictCycleSplitter):
    def _split(self, df, cont_feature_names, cat_feature_names, label_name):
        self._check_exist(df, "Material_Code", "Material_Code")
        mat_lay = np.array([str(x) for x in df["Material_Code"].copy()])
        mat_lay_set = list(sorted(set(mat_lay)))
        if "log" not in label_name and "Cycles to Failure" not in label_name:
            raise Exception(
                f"{self.__class__.__name__} requires log10(Cycles to Failure) being the target, but only {label_name} exist."
            )
        cycle = df[label_name].values.flatten()
        max_stress = df["Maximum Stress"].values.flatten()
        freq = df["Frequency"].values.flatten() if "Frequency" in df.columns else None
        r_value = df["R-value"].values.flatten() if "R-value" in df.columns else None
        min_stress = max_stress * r_value
        peak_stress = np.max(
            np.abs(np.vstack([max_stress, min_stress]).T), axis=1
        ).flatten()
        peak_stress = np.nan_to_num(peak_stress, nan=float(np.nanmax(peak_stress)))
        cycle_stress = np.linalg.norm(
            np.vstack([cycle, np.log10(peak_stress)]).T - np.array([10, 0]),
            axis=1,
        )
        inv_cycle_stress = np.max(cycle_stress) - cycle_stress

        train_indices, val_indices, test_indices = self.split_method(
            mat_lay_set,
            mat_lay,
            inv_cycle_stress,
            self.train_val_test,
            freq=freq,
            r_value=r_value,
        )

        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)

        return np.array(train_indices), np.array(val_indices), np.array(test_indices)


class RatioCycleSplitter(CycleSplitter):
    """
    Compared to CycleSplitter, only R-value is considered instead of the combination of R-value and frequency.
    """

    @classmethod
    def split_method(cls, *args, **kwargs):
        kwargs["freq"] = None
        return super(RatioCycleSplitter, cls).split_method(*args, **kwargs)


mapping = {}
clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for name, cls in clsmembers:
    if issubclass(cls, AbstractSplitter):
        mapping[name] = cls

tabensemb.data.datasplitter.splitter_mapping.update(mapping)
