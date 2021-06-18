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

        """
        Scale features using statistics that are robust to outliers.
        This Scaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range).
        The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).
        Standardization of a dataset is a common requirement for many machine learning estimators.
        Typically this is done by removing the mean and scaling to unit variance.
        However, outliers can often influence the sample mean / variance in a negative way.
        In such cases, the median and the interquartile range often give better results.
        
        If `list_of_cols` is not provided, the strategy will be applied to all numeric columns.
        If a list of columns is provided use the list, otherwise use arguments.
        For more info please see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        with_centering : boolean, True by default
            If True, center the data before scaling.
            This will cause transform to raise an exception when attempted on sparse matrices,
            because centering them entails building a dense matrix which in common use cases is likely to be too large to fit in memory.
        
        with_scaling : boolean, True by default
            If True, scale the data to interquartile range.
        quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0
            Default: (25.0, 75.0) = (1st quantile, 3rd quantile) = IQR Quantile range used to calculate scale_.
        robust_params : dict, optional
            Parmaters to pass into MinMaxScaler() constructor from Scikit-Learn
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.normalize_quantile_range('col1')
        >>> data.normalize_quantile_range(['col1', 'col2'])
        """

        list_of_cols = _input_columns(list_args, list_of_cols)

        self.train_data, self.test_data = numeric.scale(
            x_train=self.train_data,
            x_test=self.test_data,
            list_of_cols=list_of_cols,
            method="robust",
            **robust_params,
        )

        return self

    def normalize_log(self, *list_args, list_of_cols=[], base=1):

        """
        Scales data logarithmically.
        Options are 1 for natural log, 2 for base2, 10 for base10.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        base : str, optional
            Base to logarithmically scale by, by default ''
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.normalize_log('col1')
        >>> data.normalize_log(['col1', 'col2'], base=10)
        """

        list_of_cols = _input_columns(list_args, list_of_cols)

        list_of_cols = _numeric_input_conditions(list_of_cols, self.x_train)

        if not base:
            log = np.log
        elif base == 2:
            log = np.log2
        elif base == 10:
            log = np.log10
        else:
            log = np.log

        for col in list_of_cols:
            self.x_train[col] = log(self.x_train[col])

            if self.x_test is not None:
                self.x_test[col] = log(self.x_test[col])

        return self


    def split_sentences(self, *list_args, list_of_cols=[], new_col_name="_sentences"):

        """
        Splits text data into sentences and saves it into another column for analysis.
        If a list of columns is provided use the list, otherwise use arguments.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_sentences`
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.split_sentences('col1')
        >>> data.split_sentences(['col1', 'col2'])
        """

        list_of_cols = _input_columns(list_args, list_of_cols)

        for col in list_of_cols:
            if new_col_name.startswith("_"):
                new_col_name = col + new_col_name

            self.x_train[new_col_name] = pd.Series(
                map(sent_tokenize, self.x_train[col])
            )
            if self.x_test is not None:
                self.x_test[new_col_name] = pd.Series(
                    map(sent_tokenize, self.x_test[col])
                )

        return self

    def stem_nltk(
        self, *list_args, list_of_cols=[], stemmer="porter", new_col_name="_stemmed"
    ):
        """
        Transforms text to their word stem, base or root form. 
        For example:
            dogs --> dog
            churches --> church
            abaci --> abacus
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        stemmer : str, optional
            Type of NLTK stemmer to use, by default porter
            Current stemming implementations:
                - porter
                - snowball
            For more information please refer to the NLTK stemming api https://www.nltk.org/api/nltk.stem.html
        new_col_name : str, optional
            New column name to be created when applying this technique, by default `COLUMN_stemmed`
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.stem_nltk('col1')
        >>> data.stem_nltk(['col1', 'col2'], stemmer='snowball')
        """

        list_of_cols = _input_columns(list_args, list_of_cols)
        stem = NLTK_STEMMERS[stemmer]
        # Create partial for speed purposes
        func = partial(self._apply_text_method, transformer=stem.stem)

        for col in list_of_cols:
            if new_col_name.startswith("_"):
                new_col_name = col + new_col_name

            self.x_train[new_col_name] = pd.Series(map(func, self.x_train[col]))

            if self.x_test is not None:
                self.x_test[new_col_name] = pd.Series(map(func, self.x_test[col]))

        return self

    def split_words_nltk(
        self, *list_args, list_of_cols=[], regexp="", new_col_name="_tokenized"
    ):


        list_of_cols = _input_columns(list_args, list_of_cols)

        tokenizer = RegexpTokenizer(regexp)

        for col in list_of_cols:
            if new_col_name.startswith("_"):
                new_col_name = col + new_col_name

            if not regexp:
                self.x_train[new_col_name] = pd.Series(
                    map(word_tokenize, self.x_train[col])
                )

                if self.x_test is not None:
                    self.x_test[new_col_name] = pd.Series(
                        map(word_tokenize, self.x_test[col])
                    )
            else:
                self.x_train[new_col_name] = pd.Series(
                    map(tokenizer.tokenize, self.x_train[col])
                )

                if self.x_test is not None:
                    self.x_test[new_col_name] = pd.Series(
                        map(tokenizer.tokenize, self.x_test[col])
                    )

        return self
