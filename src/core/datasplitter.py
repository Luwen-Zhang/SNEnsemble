from sklearn.model_selection import train_test_split
import numpy as np
import sys, inspect


class AbstractSplitter:
    def __init__(self, train_val_test=None):
        self.train_val_test = (
            np.array([0.6, 0.2, 0.2])
            if train_val_test is None
            else np.array(train_val_test)
        )

    def split(self, df, cont_feature_names, cat_feature_names, label_name):
        train_indices, val_indices, test_indices = self._split(
            df, cont_feature_names, cat_feature_names, label_name
        )
        self._check_split(train_indices, val_indices, test_indices)
        return train_indices, val_indices, test_indices

    def _split(self, df, cont_feature_names, cat_feature_names, label_name):
        raise NotImplementedError

    def _check_split(self, train_indices, val_indices, test_indices):
        def individual_check(indices, name):
            if not issubclass(type(indices), np.ndarray):
                raise Exception(
                    f"The class of {name}_indices {type(indices)} is not the subclass of numpy.ndarray."
                )
            if len(indices.shape) != 1:
                raise Exception(
                    f"{name}_indices is not one dimensional. Use numpy.ndarray.flatten() to convert."
                )

        def intersect_check(a_indices, b_indices, a_name, b_name):
            if len(np.intersect1d(a_indices, b_indices)) != 0:
                raise Exception(
                    f"There exists intersection {np.intersect1d(a_indices, b_indices)} between {a_name}_indices "
                    f"and {b_name}_indices."
                )

        individual_check(train_indices, "train")
        individual_check(val_indices, "val")
        individual_check(test_indices, "test")

        intersect_check(train_indices, val_indices, "train", "val")
        intersect_check(train_indices, test_indices, "train", "test")
        intersect_check(val_indices, test_indices, "val", "test")

    def _check_exist(self, df, arg, name):
        if arg not in df.columns:
            raise Exception(
                f"Splitter: {name} is not a valid column in df for splitter {self.__class__.__name__}."
            )


class RandomSplitter(AbstractSplitter):
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


class MaterialSplitter(AbstractSplitter):
    def __int__(self, train_val_test=None):
        super(MaterialSplitter, self).__init__(train_val_test)

    def _split(self, df, cont_feature_names, cat_feature_names, label_name):
        self._check_exist(df, "Material_Code", "Material_Code")
        mat_lay = np.array([str(x) for x in df["Material_Code"].copy()])
        mat_lay_set = list(sorted(set(mat_lay)))

        def mat_lay_index(chosen_mat_lay, mat_lay):
            index = []
            for material in chosen_mat_lay:
                where_material = np.where(mat_lay == material)[0]
                index += list(where_material)
            return np.array(index)

        train_mat_lay, test_mat_lay = train_test_split(
            mat_lay_set, test_size=self.train_val_test[2], shuffle=True
        )
        train_mat_lay, val_mat_lay = train_test_split(
            train_mat_lay,
            test_size=self.train_val_test[1] / np.sum(self.train_val_test[0:2]),
            shuffle=True,
        )

        return (
            mat_lay_index(train_mat_lay, mat_lay),
            mat_lay_index(val_mat_lay, mat_lay),
            mat_lay_index(test_mat_lay, mat_lay),
        )


class MaterialCycleSplitter(AbstractSplitter):
    def __int__(self, train_val_test=None):
        super(MaterialCycleSplitter, self).__init__(train_val_test)

    def _split(self, df, cont_feature_names, cat_feature_names, label_name):
        self._check_exist(df, "Material_Code", "Material_Code")
        mat_lay = np.array([str(x) for x in df["Material_Code"].copy()])
        mat_lay_set = list(sorted(set(mat_lay)))
        cycle = df[label_name].values.flatten()

        train_indices = []
        val_indices = []
        test_indices = []

        for material in mat_lay_set:
            where_material = np.where(mat_lay == material)[0]
            material_cycle = cycle[where_material]
            m_train_indices = where_material[
                material_cycle
                <= np.percentile(material_cycle, np.sum(self.train_val_test[0:2]) * 100)
            ]
            m_test_indices = where_material[
                material_cycle
                > np.percentile(material_cycle, np.sum(self.train_val_test[0:2]) * 100)
            ]
            m_train_indices, m_val_indices = (
                train_test_split(
                    m_train_indices,
                    test_size=self.train_val_test[1] / np.sum(self.train_val_test[0:2]),
                    shuffle=True,
                )
                if len(m_train_indices) > 1
                else (m_train_indices, [])
            )

            train_indices += list(m_train_indices)
            val_indices += list(m_val_indices)
            test_indices += list(m_test_indices)

        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)

        return np.array(train_indices), np.array(val_indices), np.array(test_indices)


class CycleSplitter(AbstractSplitter):
    def __int__(self, train_val_test=None):
        super(CycleSplitter, self).__init__(train_val_test)

    def _split(self, df, cont_feature_names, cat_feature_names, label_name):
        cycle = df[label_name].values.flatten()

        train_indices = np.where(
            cycle <= np.percentile(cycle, np.sum(self.train_val_test[0:2]) * 100)
        )[0]
        test_indices = np.where(
            cycle > np.percentile(cycle, np.sum(self.train_val_test[0:2]) * 100)
        )[0]
        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=self.train_val_test[1] / np.sum(self.train_val_test[0:2]),
            shuffle=True,
        )

        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)

        return train_indices.flatten(), val_indices.flatten(), test_indices.flatten()


splitter_mapping = {}
clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for name, cls in clsmembers:
    if issubclass(cls, AbstractSplitter):
        splitter_mapping[name] = cls


def get_data_splitter(name: str):
    if name not in splitter_mapping.keys():
        raise Exception(f"Data splitter {name} not implemented.")
    elif not issubclass(splitter_mapping[name], AbstractSplitter):
        raise Exception(f"{name} is not the subclass of AbstractSplitter.")
    else:
        return splitter_mapping[name]
