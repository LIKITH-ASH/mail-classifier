import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class Chainer:
    encoder = None
    DELIMITER = ' -- '
    def __init__(self, columns_to_chain ):
        self.encoder = OneHotEncoder(sparse_output=False)
        self.columns_to_chain = columns_to_chain

    def linkMultipleTargetVar(self, df):
            data_encoder = self.encoder.fit_transform(df[self.columns_to_chain])

            # combining encoded data as a list or tuple
            df['y'] = [tuple(row) for row in data_encoder]
            df['y'] = df['y'].apply(lambda t: ''.join(str(int(i)) for i in t))
            return df

    def decode_unchained(self, y):

        vfunc = np.vectorize(lambda s: [np.float64(i) for i in s], otypes=[list])
        y = vfunc(y)

        data_decoder = self.encoder.inverse_transform([list(row) for row in y])
        Final_data = np.array([self.DELIMITER.join(map(str, row)) for row in data_decoder])

        return Final_data

    def remove_type(self, y: np.ndarray, num_types: int = 1):
        y = pd.Series(y)
        return y.apply(lambda x: Chainer.DELIMITER.join(x.split(Chainer.DELIMITER)[:-num_types]))