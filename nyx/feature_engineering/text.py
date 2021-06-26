
import pandas as pd
import spacy

from textblob import TextBlob

from nyx.util import _get_columns

def textblob_features(
    x_train, x_test, feature, list_of_cols=[], new_col_name="_postagged",
):

    """
    Part of Speech tag the text data provided. Used to tag each word as a Noun, Adjective,
    Verbs, etc.
    This utilizes TextBlob which utlizes the NLTK tagger and is a wrapper for the tagging process.
    
    Parameters
    ----------
    x_train : DataFrame
        Dataset
    x_test : DataFrame
        Testing dataset, by default None
    feature : str,
        Textblob feature
    list_of_cols : list, optional
        A list of specific columns to apply this technique to, by default []
    new_col_name : str, optional
        New column name to be created when applying this technique, by default `COLUMN_postagged`
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column.
    Returns 2 Dataframes if x_test is provided. 
    """

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