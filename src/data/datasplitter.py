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


class CycleSplitter(AbstractSplitter):
    """
    Split the dataset by the material code and the number of cycles to simulate the scenario of prediction using
    limited data from accelerated fatigue tests. Validation/testing sets contain data that have larger number of cycles
    than the training set (even much larger in the testing set). If given at least one of "Frequency" and "R-value",
    the splitting is performed for each material and for each combination of frequency-R (instead of only for each
    material).

    Steps of the splitting are (assuming a 60:20:20 split):
    1. Find data points for each material and for each combination of frequency-R.
    2. In these data points, find 60% points with lower Nf for the training set, and the rest of them are for validation
       and testing.
    3. In the 40% part, find 50% points with lower Nf for the validation set, and the rest of them are for testing.
    4. To simulation the real scenario and to prevent over-fitting on the validation set, the training and validation
       sets are completely shuffled.
    """

    def _split(self, df, cont_feature_names, cat_feature_names, label_name):
        self._check_exist(df, "Material_Code", "Material_Code")
        mat_lay = np.array([str(x) for x in df["Material_Code"].copy()])
        mat_lay_set = list(sorted(set(mat_lay)))
        cycle = df[label_name[0]].values.flatten()
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

        train_indices, val_indices, test_indices = cls._after_split(
            train_indices, val_indices, test_indices, train_val_test
        )
        return np.array(train_indices), np.array(val_indices), np.array(test_indices)

    @classmethod
    def _after_split(cls, train_indices, val_indices, test_indices, train_val_test):
        train_val_indices = np.concatenate([train_indices, val_indices])
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=train_val_test[1] / np.sum(train_val_test[:2]),
            shuffle=True,
        )
        return train_indices, val_indices, test_indices

    @classmethod
    def _split_one_fr(cls, where_fr, fr_cycle, train_val_test):
        find_closest = lambda x, arr: arr[
            np.where(np.abs(arr - x) == np.min(np.abs(arr - x)))[0][0]
        ]
        percentile_closest_point = lambda arr, p: find_closest(
            np.percentile(arr, p), arr
        )

        # Values less (or less/equal) than the value closest to the percentile is selected to be in the training set.
        # Whether the value closest to the percentile will be in the training set is determined by the ratio of the
        # training set.
        def percentile_split(arr, indices, r1, r2):
            if len(arr) == 0:
                return np.array([], dtype=int), np.array([], dtype=int)
            l_symbol = (
                np.less_equal
                if np.random.choice([0, 1], p=(r1 / (r1 + r2), r2 / (r1 + r2))) == 0
                else np.less
            )
            first_indices = indices[
                l_symbol(arr, percentile_closest_point(arr, r1 / (r1 + r2) * 100))
            ]
            second_indices = np.setdiff1d(indices, first_indices)
            return first_indices, second_indices

        fr_train_indices, fr_test_val_indices = percentile_split(
            fr_cycle,
            np.arange(len(fr_cycle)),
            train_val_test[0],
            sum(train_val_test[1:]),
        )
        fr_val_indices, fr_test_indices = percentile_split(
            fr_cycle[fr_test_val_indices],
            fr_test_val_indices,
            train_val_test[1],
            train_val_test[2],
        )
        return (
            where_fr[fr_train_indices],
            where_fr[fr_val_indices],
            where_fr[fr_test_indices],
        )


class StressCycleSplitter(CycleSplitter):
    def _split(self, df, cont_feature_names, cat_feature_names, label_name):
        self._check_exist(df, "Material_Code", "Material_Code")
        mat_lay = np.array([str(x) for x in df["Material_Code"].copy()])
        mat_lay_set = list(sorted(set(mat_lay)))
        cycle = df[label_name[0]].values.flatten()
        max_stress = df["Maximum Stress"].values.flatten()
        freq = df["Frequency"].values.flatten() if "Frequency" in df.columns else None
        r_value = df["R-value"].values.flatten() if "R-value" in df.columns else None
        min_stress = max_stress * r_value
        peak_stress = np.max(
            np.abs(np.vstack([max_stress, min_stress]).T), axis=1
        ).flatten()
        log_peak_stress = np.log10(
            np.nan_to_num(peak_stress, nan=float(np.nanmax(peak_stress)))
        )
        norm_cycle = (cycle - np.max(cycle)) / (np.max(cycle) - np.min(cycle))
        norm_log_peak_stress = (log_peak_stress - np.max(log_peak_stress)) / (
            np.max(log_peak_stress) - np.min(log_peak_stress)
        )
        cycle_stress = np.linalg.norm(
            np.vstack([norm_cycle, norm_log_peak_stress]).T - np.array([1, 0]),
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


mapping = {}
clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for name, cls in clsmembers:
    if issubclass(cls, AbstractSplitter):
        mapping[name] = cls

tabensemb.data.datasplitter.splitter_mapping.update(mapping)
