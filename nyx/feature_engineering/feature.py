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

        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        keep_col : bool
            A parameter to specify whether to drop the column being transformed, by default
            keep the column, True
        categories : ‘auto’ or a list of array-like, default=’auto’
            Categories (unique values) per feature:
                ‘auto’ : Determine categories automatically from the training data.
                list : categories[i] holds the categories expected in the ith column. The passed categories should not mix strings and numeric values within a single feature, and should be sorted in case of numeric values.
            The used categories can be found in the categories_ attribute.
        drop : ‘first’ or a array-like of shape (n_features,), default=None
            Specifies a methodology to use to drop one of the categories per feature. This is useful in situations where perfectly collinear features cause problems, such as when feeding the resulting data into a neural network or an unregularized regression.
                None : retain all features (the default).
                ‘first’ : drop the first category in each feature. If only one category is present, the feature will be dropped entirely.
                array : drop[i] is the category in feature X[:, i] that should be dropped.
        sparsebool : default=True
            Will return sparse matrix if set True else will return an array.
        dtype : number type, default=np.float
            Desired dtype of output.
        handle_unknown: {‘error’, ‘ignore’}, default='ignore'
            Whether to raise an error or ignore if an unknown categorical feature is present during transform (default is to raise).
            When this parameter is set to ‘ignore’ and an unknown category is encountered during transform, the resulting one-hot encoded columns for this feature will be all zeros.
            In the inverse transform, an unknown category will be denoted as None.

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

    def tfidf(self, *list_args, list_of_cols=[], keep_col=True, **tfidf_kwargs):


        """
        Creates a matrix of the tf-idf score for every word in the corpus as it pertains to each document.
        The higher the score the more important a word is to a document, the lower the score (relative to the other scores)
        the less important a word is to a document.
        For more information see: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        If a list of columns is provided use the list, otherwise use arguments.

        """

        # If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        enc = TfidfVectorizer(**tfidf_kwargs)
        list_of_cols = _get_columns(list_of_cols, self.x_train)

        for col in list_of_cols:
            enc_data = enc.fit_transform(self.x_train[col]).toarray()
            enc_df = pd.DataFrame(enc_data, columns=enc.get_feature_names())
            self.x_train = drop_replace_columns(self.x_train, col, enc_df, keep_col)

            if self.x_test is not None:
                enc_test = enc.transform(self.x_test[col]).toarray()
                enc_test_df = pd.DataFrame(enc_test, columns=enc.get_feature_names())
                self.x_test = drop_replace_columns(
                    self.x_test, col, enc_test_df, keep_col
                )

        return self
