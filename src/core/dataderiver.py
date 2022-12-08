import numpy as np


class AbstractDeriver:
    def __init__(self):
        pass

    def derive(self, df, derived_name, col_names, stacked, intermediate=False, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _generate_col_names(derived_name, length, col_names):
        if col_names is None or len(col_names) != length:
            names = [f'{derived_name}-{idx}' for idx in range(length)] if length > 1 else [derived_name]
        else:
            names = col_names
        return names

    def _check_arg(self, arg, name):
        if arg is None:
            raise Exception(f'Derivation: {name} should be specified for deriver {self.__class__.__name__}')

    def _check_exist(self, df, arg, name):
        if arg not in df.columns:
            raise Exception(f'Derivation: {name} is not a valid column in df for deriver {self.__class__.__name__}.')


class DegLayerDeriver(AbstractDeriver):
    def __init__(self):
        super(DegLayerDeriver, self).__init__()

    def derive(self, df, derived_name=None, col_names=None, stacked=True, intermediate=False, sequence_column=None):
        self._check_arg(derived_name, 'derived_name')
        self._check_arg(sequence_column, 'sequence_column')
        self._check_exist(df, sequence_column, 'sequence_column')

        sequence = [[int(y) if y != 'nan' else np.nan for y in str(x).split('/')] for x in
                    df['Sequence'].values]

        deg_layers = np.zeros((len(sequence), 4),
                              dtype=np.int)  # for 0-deg, pm45-deg, 90-deg, and other directions respectively

        for idx, seq in enumerate(sequence):
            deg_layers[idx, 0] = seq.count(0)
            deg_layers[idx, 1] = seq.count(45) + seq.count(-45)
            deg_layers[idx, 2] = seq.count(90)
            deg_layers[idx, 3] = len(seq) - seq.count(np.nan) - np.sum(deg_layers[idx, :3])

        names = self._generate_col_names(derived_name, deg_layers.shape[1], col_names)

        related_columns = [sequence_column]

        return deg_layers, derived_name, names, stacked, intermediate, related_columns


class RelativeDeriver(AbstractDeriver):
    def __init__(self):
        super(RelativeDeriver, self).__init__()

    def derive(self, df, derived_name=None, col_names=None, stacked=True, intermediate=False, absolute_col=None, relative2_col=None):
        self._check_arg(derived_name, 'derived_name')
        self._check_arg(absolute_col, 'absolute_col')
        self._check_arg(relative2_col, 'relative2_col')
        self._check_exist(df, absolute_col, 'absolute_col')
        self._check_exist(df, relative2_col, 'relative2_col')

        relative = df[absolute_col] / df[relative2_col]
        relative = relative.values

        names = self._generate_col_names(derived_name, 1, col_names)
        related_columns = [absolute_col, relative2_col]

        return relative, derived_name, names, stacked, intermediate, related_columns


class SuppStressDeriver(AbstractDeriver):
    def __init__(self):
        super(SuppStressDeriver, self).__init__()

    def derive(self, df, derived_name=None, col_names=None, stacked=True, intermediate=False, max_stress_col=None, min_stress_col=None, relative=False):
        self._check_arg(derived_name, 'derived_name')
        self._check_arg(max_stress_col, 'max_stress_col')
        self._check_arg(min_stress_col, 'min_stress_col')
        self._check_exist(df, max_stress_col, 'max_stress_col')
        self._check_exist(df, min_stress_col, 'min_stress_col')

        df_tmp = df.copy()

        df_tmp['Absolute Maximum Stress'] = np.nan
        df_tmp['Absolute Peak-to-peak Stress'] = np.nan
        df_tmp['Absolute Mean Stress'] = np.nan
        df_tmp['Relative Maximum Stress'] = np.nan
        df_tmp['Relative Peak-to-peak Stress'] = np.nan
        df_tmp['Relative Mean Stress'] = np.nan

        for idx in df_tmp.index:
            s = np.array([df_tmp.loc[idx, max_stress_col], df_tmp.loc[idx, min_stress_col]])
            which_max_stress = np.where(np.abs(s) == np.max(np.abs(s)))[0]
            if len(which_max_stress) == 0:
                which_max_stress = 1 - int(np.isnan(s[1]))  # when nan appears in s
            else:
                which_max_stress = which_max_stress[0]

            relative_to = np.abs(df_tmp.loc[idx, 'Static Maximum Tensile Stress']) \
                if s[which_max_stress] > 0 else np.abs(df_tmp.loc[idx, 'Static Maximum Compressive Stress'])
            if np.isnan(relative_to) and s[0] + s[1] < 1e-5 and s[which_max_stress] > 0:
                relative_to = np.abs(df_tmp.loc[idx, 'Static Maximum Compressive Stress'])

            df_tmp.loc[idx, 'Absolute Maximum Stress'] = s[which_max_stress]
            p2p = np.abs(s[0] - s[1])
            if np.isnan(p2p):
                p2p = np.abs(s[1 - int(np.isnan(s[1]))])
            df_tmp.loc[idx, 'Absolute Peak-to-peak Stress'] = p2p
            df_tmp.loc[idx, 'Absolute Mean Stress'] = s[which_max_stress] - np.sign(s[which_max_stress]) * p2p / 2

            if np.abs(s[which_max_stress] / relative_to) <= 1.1:  # otherwise static data is not correct
                df_tmp.loc[idx, 'Relative Maximum Stress'] = np.abs(s[which_max_stress] / relative_to)
                df_tmp.loc[idx, 'Relative Peak-to-peak Stress'] = np.abs(p2p / relative_to)
                df_tmp.loc[idx, 'Relative Mean Stress'] = np.abs(df_tmp.loc[idx, 'Absolute Mean Stress'] / relative_to)
            else:
                df_tmp.loc[idx, 'Static Maximum Tensile Stress'] = np.nan
                df_tmp.loc[idx, 'Static Maximum Compressive Stress'] = np.nan

        names = ['Absolute Maximum Stress', 'Absolute Peak-to-peak Stress', 'Absolute Mean Stress',
                 'Relative Maximum Stress', 'Relative Peak-to-peak Stress', 'Relative Mean Stress'] \
            if relative else ['Absolute Maximum Stress', 'Absolute Peak-to-peak Stress', 'Absolute Mean Stress']
        stresses = df_tmp[names].values
        related_columns = [max_stress_col, min_stress_col]

        return stresses, derived_name, names, stacked, intermediate, related_columns


deriver_mapping = {
    'DegLayerDeriver': DegLayerDeriver(),
    'RelativeDeriver': RelativeDeriver(),
    'SuppStressDeriver': SuppStressDeriver(),
}


def get_data_deriver(name: str):
    if name not in deriver_mapping.keys():
        raise Exception(f'Data deriver {name} not implemented or added to dataprocessor.processor_mapping.')
    elif not issubclass(type(deriver_mapping[name]), AbstractDeriver):
        raise Exception(f'{name} is not the subclass of AbstractDeriver.')
    else:
        return deriver_mapping[name]
