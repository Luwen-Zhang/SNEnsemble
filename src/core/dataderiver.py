import numpy as np

class AbstractDeriver:
    def __init__(self):
        pass

    def derive(self, df, **kargs):
        raise NotImplementedError

class DegLayerDeriver(AbstractDeriver):
    def __init__(self):
        super(DegLayerDeriver, self).__init__()

    def derive(self, df, sequence_column='Sequence', derived_name='deg_layers'):
        if sequence_column not in df.columns:
            raise Exception(f'Derivation: {sequence_column} is not a valid column.')

        sequence = [[int(y) if y != 'nan' else np.nan for y in str(x).split('/')] for x in
                    df['Sequence'].values]

        deg_layers = np.zeros((len(sequence), 4),
                              dtype=np.int)  # for 0-deg, pm45-deg, 90-deg, and other directions respectively

        for idx, seq in enumerate(sequence):
            deg_layers[idx, 0] = seq.count(0)
            deg_layers[idx, 1] = seq.count(45) + seq.count(-45)
            deg_layers[idx, 2] = seq.count(90)
            deg_layers[idx, 3] = len(seq) - seq.count(np.nan) - np.sum(deg_layers[idx, :3])

        return deg_layers, derived_name

deriver_mapping = {
    'DegLayerDeriver': DegLayerDeriver(),
}


def get_data_deriver(name: str):
    if name not in deriver_mapping.keys():
        raise Exception(f'Data deriver {name} not implemented or added to dataprocessor.processor_mapping.')
    elif not issubclass(type(deriver_mapping[name]), AbstractDeriver):
        raise Exception(f'{name} is not the subclass of AbstractDeriver.')
    else:
        return deriver_mapping[name]