def process_text(
    corpus, lower=True, punctuation=True, stopwords=True, stemmer=True, numbers=True,
):

    import nltk
    import string
    from nltk.stem.snowball import SnowballStemmer
    from nltk.tokenize import word_tokenize

    transformed_corpus = ""

    if lower:
        corpus = corpus.lower()

    for token in word_tokenize(corpus):

        if punctuation:
            if token in string.punctuation:
                continue

            token = token.translate(str.maketrans("", "", string.punctuation))

        if numbers:
            token = token.translate(str.maketrans("", "", "0123456789"))

        if stopwords:
            stop_words = nltk.corpus.stopwords.words("english")
            if token in stop_words:
                continue

        if stemmer:
            stem = SnowballStemmer("english")
            token = stem.stem(token)

        transformed_corpus += token + " "

    return transformed_corpus.strip()