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


        list_of_cols = _input_columns(list_args, list_of_cols)

        self.train_data, self.test_data = numeric.scale(
            x_train=self.train_data,
            x_test=self.test_data,
            list_of_cols=list_of_cols,
            method="minmax",
            **normalize_params,
        )

        return self