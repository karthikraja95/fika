import gensim
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.summarization import keywords
from gensim.summarization.summarizer import summarize
from nltk.tokenize import word_tokenize

from nyx.preprocessing.text import process_text

def gensim_textrank_summarizer(
    x_train, x_test=None, list_of_cols=[], new_col_name="_summarized", **algo_kwargs
):
    """
    Uses Gensim Text Rank summarize to extractively summarize text.
    Note this uses a variant of Text Rank.
    
    Parameters
    ----------
    x_train : DataFrame
        Dataset
    x_test : DataFrame
        Testing dataset, by default None
    list_of_cols : list, optional
        Column name(s) of text data that you want to summarize
    new_col_name : str, optional
        New column name to be created when applying this technique, by default `_summarized`
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column.
    Returns 2 Dataframes if x_test is provided. 
    """

    for col in list_of_cols:
        if new_col_name.startswith("_"):
            new_col_name = col + new_col_name

        x_train.loc[:, new_col_name] = [
            summarize(x, **algo_kwargs) for x in x_train[col]
        ]

        if x_test is not None:
            x_test.loc[:, new_col_name] = [
                summarize(x, **algo_kwargs) for x in x_test[col]
            ]

    return x_train, x_test

    def gensim_textrank_keywords(
    x_train,
    x_test=None,
    list_of_cols=[],
    new_col_name="_extracted_keywords",
    **algo_kwargs
):
    """
    Uses Gensim Text Rank summarize to extract keywords.
    Note this uses a variant of Text Rank.
    
    Parameters
    ----------
    x_train : DataFrame
        Dataset
    x_test : DataFrame
        Testing dataset, by default None
    list_of_cols : list, optional
        Column name(s) of text data that you want to summarize
    new_col_name : str, optional
        New column name to be created when applying this technique, by default `_extracted_keywords`
    
    Returns
    -------
    Dataframe, *Dataframe
        Transformed dataframe with the new column.
    Returns 2 Dataframes if x_test is provided. 
    """

    for col in list_of_cols:

        if new_col_name.startswith("_"):
            new_col_name = col + new_col_name

        x_train.loc[:, new_col_name] = [
            keywords(x, **algo_kwargs) for x in x_train[col]
        ]

        if x_test is not None:
            x_test.loc[:, new_col_name] = [
                keywords(x, **algo_kwargs) for x in x_test[col]
            ]

    return x_train, x_test

def gensim_word2vec(x_train, x_test=None, prep=False, col_name=None, **algo_kwargs):
    """
    Uses Gensim Text Rank summarize to extract keywords.
    Note this uses a variant of Text Rank.
    
    Parameters
    ----------
    x_train : DataFrame
        Dataset
    x_test : DataFrame
        Testing dataset, by default None
    prep : bool, optional
        True to prep the text
        False if text is already prepped.
        By default, False
    col_name : str, optional
        Column name of text data that you want to summarize
        
    Returns
    -------
    Word2Vec
        Word2Vec model
    """

    if prep:
        w2v = Word2Vec(
            sentences=[word_tokenize(process_text(text)) for text in x_train[col_name]],
            **algo_kwargs
        )
    else:
        w2v = Word2Vec(sentences=x_train[col_name], **algo_kwargs)

    return w2v

