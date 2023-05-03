import numpy as np
from src.utils import *
from src.data import AbstractSplitter
import inspect
from sklearn.model_selection import train_test_split
from typing import Type


class RandomSplitter(AbstractSplitter):
    """
    Randomly split the dataset.
    """

    def __int__(self, train_val_test=None):
        super(RandomSplitter, self).__init__(train_val_test)

    def _split(self, df, cont_feature_names, cat_feature_names, label_name):
        length = len(df)
        train_indices, test_indices = train_test_split(
            np.arange(length), test_size=self.train_val_test[2], shuffle=True
        )
        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=self.train_val_test[1] / np.sum(self.train_val_test[0:2]),
            shuffle=True,
        )

        return train_indices, val_indices, test_indices


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

    def __init__(self, train_val_test=None):
        super(MaterialSplitter, self).__init__(train_val_test)

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


class MaterialCycleSplitter(AbstractSplitter):
    """
    Split the dataset by the material code and the number of cycles to simulate the scenario of designing new materials
    using limited data from accelerated fatigue tests. Training/validation/testing datasets will contain entirely
    different materials, and validation/testing sets contain data that generally have larger number of cycles than the
    training set (even much larger in the testing set).
    """

    def __init__(self, train_val_test=None):
        super(MaterialCycleSplitter, self).__init__(train_val_test)

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
    """

    def __init__(self, train_val_test=None):
        super(CycleSplitter, self).__init__(train_val_test)

    def _split(self, df, cont_feature_names, cat_feature_names, label_name):
        self._check_exist(df, "Material_Code", "Material_Code")
        mat_lay = np.array([str(x) for x in df["Material_Code"].copy()])
        mat_lay_set = list(sorted(set(mat_lay)))
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
                for fr in set(material_freq_r):
                    where_fr = np.where(material_freq_r == fr)[0]
                    fr_cycle = material_cycle[where_fr]
                    fr_train_indices = where_fr[
                        fr_cycle <= np.percentile(fr_cycle, train_val_test[0] * 100)
                    ]
                    fr_test_indices = where_fr[
                        fr_cycle
                        > np.percentile(fr_cycle, np.sum(train_val_test[0:2]) * 100)
                    ]
                    fr_val_indices = np.setdiff1d(
                        where_fr, np.append(fr_train_indices, fr_test_indices)
                    )
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

        return np.array(train_indices), np.array(val_indices), np.array(test_indices)


splitter_mapping = {}
clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for name, cls in clsmembers:
    if issubclass(cls, AbstractSplitter):
        splitter_mapping[name] = cls


def get_data_splitter(name: str) -> Type[AbstractSplitter]:
    if name not in splitter_mapping.keys():
        raise Exception(f"Data splitter {name} not implemented.")
    elif not issubclass(splitter_mapping[name], AbstractSplitter):
        raise Exception(f"{name} is not the subclass of AbstractSplitter.")
    else:
        return splitter_mapping[name]
