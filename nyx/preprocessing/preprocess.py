import string
import pandas as pd
import numpy as np

from functools import partial
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import PorterStemmer, SnowballStemmer
from nltk.tokenize import RegexpTokenizer, word_tokenize

from nyx.preprocessing import numeric, text

from nyx.util import (
    _input_columns,
    _numeric_input_conditions,
)

NLTK_STEMMERS = {"porter": PorterStemmer(), "snowball": SnowballStemmer("english")}

NLTK_LEMMATIZERS = {"wordnet": WordNetLemmatizer()}


class Preprocess(object):

    def normalize_numeric(self, *list_args, list_of_cols=[], **normalize_params):

        """
        Function that normalizes all numeric values between 2 values to bring features into same domain.
        
        If `list_of_cols` is not provided, the strategy will be applied to all numeric columns.
        If a list of columns is provided use the list, otherwise use arguments.
        For more info please see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler 
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        feature_range : tuple(int or float, int or float), optional
            Min and max range to normalize values to, by default (0, 1)
        normalize_params : dict, optional
            Parmaters to pass into MinMaxScaler() constructor from Scikit-Learn
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.normalize_numeric('col1')
        >>> data.normalize_numeric(['col1', 'col2'])
        """
        
        list_of_cols = _input_columns(list_args, list_of_cols)

        self.train_data, self.test_data = numeric.scale(
            x_train=self.train_data,
            x_test=self.test_data,
            list_of_cols=list_of_cols,
            method="minmax",
            **normalize_params,
        )

        return self

    def normalize_quantile_range(self, *list_args, list_of_cols=[], **robust_params):

        list_of_cols = _input_columns(list_args, list_of_cols)

        self.train_data, self.test_data = numeric.scale(
            x_train=self.train_data,
            x_test=self.test_data,
            list_of_cols=list_of_cols,
            method="robust",
            **robust_params,
        )

        return self

