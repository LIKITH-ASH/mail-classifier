import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class Chainer:
    encoder = None
    DELIMITER = ' -- '
    def __init__(self, columns_to_chain ):
        self.encoder = OneHotEncoder(sparse_output=False)
        self.columns_to_chain = columns_to_chain
