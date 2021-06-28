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

        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        keep_col : bool, optional
            True if you want to keep the column(s) or False if you want to drop the column(s)
        encoding: str, default=’utf-8’
            If bytes or files are given to analyze, this encoding is used to decode.
        decode_error: {‘strict’, ‘ignore’, ‘replace’} (default=’strict’)
            Instruction on what to do if a byte sequence is given to analyze that contains characters not of the given encoding.
            By default, it is ‘strict’, meaning that a UnicodeDecodeError will be raised.
            Other values are ‘ignore’ and ‘replace’.
        strip_accents: {‘ascii’, ‘unicode’, None} (default=None)
            Remove accents and perform other character normalization during the preprocessing step.\
            ‘ascii’ is a fast method that only works on characters that have an direct ASCII mapping.
            ‘unicode’ is a slightly slower method that works on any characters.
            None (default) does nothing.
            Both ‘ascii’ and ‘unicode’ use NFKD normalization from unicodedata.normalize.
        lowercase: bool (default=True)
            Convert all characters to lowercase before tokenizing.
        preprocessor: callable or None (default=None)
            Override the preprocessing (string transformation) stage while preserving the tokenizing and n-grams generation steps.
            Only applies if analyzer is not callable.
        tokenizer: callable or None (default=None)
            Override the string tokenization step while preserving the preprocessing and n-grams generation steps.
            Only applies if analyzer == 'word'.
        analyzer: str, {‘word’, ‘char’, ‘char_wb’} or callable
            Whether the feature should be made of word or character n-grams
            Option ‘char_wb’ creates character n-grams only from text inside word boundaries;
            n-grams at the edges of words are padded with space.
            If a callable is passed it is used to extract the sequence of features out of the raw, unprocessed input.
        stop_words: str {‘english’}, list, or None (default=None)
            If a string, it is passed to _check_stop_list and the appropriate stop list is returned.
            ‘english’ is currently the only supported string value.
            There are several known issues with ‘english’ and you should consider an alternative (see Using stop words).
            If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens. Only applies if analyzer == 'word'.
            If None, no stop words will be used. max_df can be set to a value in the range [0.7, 1.0) to automatically detect and filter stop words based on intra corpus document frequency of terms.
        token_pattern: str
            Regular expression denoting what constitutes a “token”, only used if analyzer == 'word'.
            The default regexp selects tokens of 2 or more alphanumeric characters (punctuation is completely ignored and always treated as a token separator).
        
        ngram_range: tuple (min_n, max_n), default=(1, 1)
            The lower and upper boundary of the range of n-values for different n-grams to be extracted.
            All values of n such that min_n <= n <= max_n will be used.
            For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams.
            Only applies if analyzer is not callable.
        
        max_df: float in range [0.0, 1.0] or int (default=1.0)
            When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).
            If float, the parameter represents a proportion of documents, integer absolute counts.
            This parameter is ignored if vocabulary is not None.
        min_df: float in range [0.0, 1.0] or int (default=1)
            When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
            This value is also called cut-off in the literature.
            If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
        max_features: int or None (default=None)
            If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
            This parameter is ignored if vocabulary is not None.
        vocabulary: Mapping or iterable, optional (default=None)
            Either a Mapping (e.g., a dict) where keys are terms and values are indices in the feature matrix, or an iterable over terms.
            If not given, a vocabulary is determined from the input documents.
        binary: bool (default=False)
            If True, all non-zero term counts are set to 1.
            This does not mean outputs will have only 0/1 values, only that the tf term in tf-idf is binary. (Set idf and normalization to False to get 0/1 outputs).
        dtype: type, optional (default=float64)
            Type of the matrix returned by fit_transform() or transform().
        norm: ‘l1’, ‘l2’ or None, optional (default=’l2’)
            Each output row will have unit norm, either: * ‘l2’: Sum of squares of vector elements is 1.
            The cosine similarity between two vectors is their dot product when l2 norm has been applied. * ‘l1’: Sum of absolute values of vector elements is 1.
        use_idf: bool (default=True)
            Enable inverse-document-frequency reweighting.
        smooth_idf: bool (default=True)
            Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once.
            Prevents zero divisions.
        sublinear_tf: bool (default=False)
            Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.tfidf('col1', 'col2', 'col3')
        >>> data.tfidf('col1', 'col2', 'col3', lowercase=False, smoothidf=False)   
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

    def bag_of_words(self, *list_args, list_of_cols=[], keep_col=True, **bow_kwargs):

        """
        Creates a matrix of how many times a word appears in a document.
        The premise is that the more times a word appears the more the word represents that document.
        For more information see: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        If a list of columns is provided use the list, otherwise use arguments.
        
        """

        # If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        enc = CountVectorizer(**bow_kwargs)
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
