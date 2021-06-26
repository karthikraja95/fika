
import pandas as pd
import spacy

from textblob import TextBlob

from nyx.util import _get_columns

def textblob_features(
    x_train, x_test, feature, list_of_cols=[], new_col_name="_postagged",
):

    list_of_cols = _get_columns(list_of_cols, x_train)

    for col in list_of_cols:

        if new_col_name.startswith("_"):
            new_col_name = col + new_col_name

        x_train[new_col_name] = pd.Series(
            [getattr(TextBlob(x), feature) for x in x_train[col]]
        )

        if x_test is not None:
            x_test[new_col_name] = pd.Series(
                [getattr(TextBlob(x), feature) for x in x_test[col]]
            )

    return x_train, x_test