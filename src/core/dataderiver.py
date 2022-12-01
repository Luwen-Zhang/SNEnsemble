import numpy as np


class AbstractDeriver:
    def __init__(self):
        pass

    def derive(self, df, derived_name, col_names, stacked, **kargs):
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

    def derive(self, df, sequence_column=None, derived_name=None, col_names=None, stacked=True):
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

        return deg_layers, derived_name, names, stacked, related_columns


class MeanStressDeriver(AbstractDeriver):
    def __init__(self):
        super(MeanStressDeriver, self).__init__()

    def derive(self, df, derived_name=None, col_names=None, stacked=True, abs_maximum_col=None, p2p_col=None):
        self._check_arg(derived_name, 'derived_name')
        self._check_arg(abs_maximum_col, 'abs_maximum_col')
        self._check_arg(p2p_col, 'p2p_col')
        self._check_exist(df, abs_maximum_col, 'abs_maximum_col')
        self._check_exist(df, p2p_col, 'p2p_col')

        mean_stress = df[abs_maximum_col] - np.sign(df[abs_maximum_col]) * df[p2p_col] / 2
        mean_stress = mean_stress.values

        names = self._generate_col_names(derived_name, 1, col_names)
        related_columns = [abs_maximum_col, p2p_col]

        return mean_stress, derived_name, names, stacked, related_columns


class RelativeDeriver(AbstractDeriver):
    def __init__(self):
        super(RelativeDeriver, self).__init__()

    def derive(self, df, derived_name=None, col_names=None, stacked=True, absolute_col=None, relative2_col=None):
        self._check_arg(derived_name, 'derived_name')
        self._check_arg(absolute_col, 'absolute_col')
        self._check_arg(relative2_col, 'relative2_col')
        self._check_exist(df, absolute_col, 'absolute_col')
        self._check_exist(df, relative2_col, 'relative2_col')

        relative = df[absolute_col] / df[relative2_col]
        relative = relative.values

        names = self._generate_col_names(derived_name, 1, col_names)
        related_columns = [absolute_col, relative2_col]

        return relative, derived_name, names, stacked, related_columns


deriver_mapping = {
    'DegLayerDeriver': DegLayerDeriver(),
    'MeanStressDeriver': MeanStressDeriver(),
    'RelativeDeriver': RelativeDeriver(),
}


def get_data_deriver(name: str):
    if name not in deriver_mapping.keys():
        raise Exception(f'Data deriver {name} not implemented or added to dataprocessor.processor_mapping.')
    elif not issubclass(type(deriver_mapping[name]), AbstractDeriver):
        raise Exception(f'{name} is not the subclass of AbstractDeriver.')
    else:
        return deriver_mapping[name]
