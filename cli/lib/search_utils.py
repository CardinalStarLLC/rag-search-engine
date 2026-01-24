import json
import re
import string
from nltk.stem import PorterStemmer
from typing import Dict, List

DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DEFAULT_CHUNK_OVERLAP = 1
MAX_SEARCH_RESULTS = 5
BM25_B = 0.75
BM25_K1 = 1.5

stemmer = PorterStemmer()

def load_stop_words():
    global stop_words_list
    with open('data/stopwords.txt', 'r') as file:
        stop_words_list = file.read().splitlines()
        return stop_words_list

def load_movie_data() -> Dict[int, str]:
    movie_dict: Dict[int, str] = {}
    with open('data/movies.json', 'r') as file:
        data = json.load(file)

    for m in data['movies']:
        movie_dict[int(m['id'])] = m['title']
    
    return movie_dict

def load_movies():
    data = open_json_file('data/movies.json')
    return data['movies']

def open_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

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

def semantic_chunk(text: str, max_chunk_size: int, overlap: int = 0) -> List[str]:
    separator = " "
    sentences = split_text_to_sentences(text)
    
    chunks: List[str] = []
    index = 0
    
    while index <= len(sentences):
        chunks.append(separator.join(sentences[index:index + max_chunk_size]))
        index += max_chunk_size
        if (index >= len(sentences)):
            break

        index -= overlap

    return chunks

def split_text_to_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+", text)