import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import PolynomialFeatures

from nyx.feature_engineering import text
from nyx.feature_engineering import util
from nyx.util import (
    _input_columns,
    _get_columns,
    drop_replace_columns,
    _numeric_input_conditions,
)

class Feature(object):

    def onehot_encode(
        self, *list_args, list_of_cols=[], keep_col=True, **onehot_kwargs
    ):

        """
        Creates a matrix of converted categorical columns into binary columns of ones and zeros.
        For more info see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        If a list of columns is provided use the list, otherwise use arguments.


        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.onehot_encode('col1', 'col2', 'col3')
        >>> data.onehot_encode('col1', 'col2', 'col3', drop='first')
        
        """
        # If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        enc = OneHotEncoder(handle_unknown="ignore", **onehot_kwargs)
        list_of_cols = _get_columns(list_of_cols, self.x_train)

        enc_data = enc.fit_transform(self.x_train[list_of_cols]).toarray()
        enc_df = pd.DataFrame(enc_data, columns=enc.get_feature_names(list_of_cols))
        self.x_train = drop_replace_columns(
            self.x_train, list_of_cols, enc_df, keep_col
        )

        if self.x_test is not None:
            enc_test = enc.transform(self.x_test[list_of_cols]).toarray()
            enc_test_df = pd.DataFrame(
                enc_test, columns=enc.get_feature_names(list_of_cols)
            )
            self.x_test = drop_replace_columns(
                self.x_test, list_of_cols, enc_test_df, keep_col
            )

        return self
