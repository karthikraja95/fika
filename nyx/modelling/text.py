import gensim
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.summarization import keywords
from gensim.summarization.summarizer import summarize
from nltk.tokenize import word_tokenize

from nyx.preprocessing.text import process_text