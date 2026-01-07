import string
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

MAX_SEARCH_RESULTS = 5
BM25_B = 0.75
BM25_K1 = 1.5

def load_stop_words():
    global stop_words_list
    with open('data/stopwords.txt', 'r') as file:
        stop_words_list = file.read().splitlines()
        return stop_words_list

def tokenize(text: str):
    translation_table = str.maketrans("", "", string.punctuation)

    tokens = text.lower().translate(translation_table).split()
    valid_tokens = []
    for token in tokens:
        if token in stop_words_list:
            continue
        stem_word = stemmer.stem(token)
        valid_tokens.append(stem_word)
    return valid_tokens