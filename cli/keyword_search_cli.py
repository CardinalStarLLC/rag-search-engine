#!/usr/bin/env python3

import argparse
import json
import string
from nltk.stem import PorterStemmer
# imports for InvertedIndex
from typing import List, Dict, Any, Set, Iterable
from collections import Counter
import pickle
import os

stemmer = PorterStemmer()

def load_stop_words():
    global stop_words_list
    with open('data/stopwords.txt', 'r') as file:
        stop_words_list = file.read().splitlines()
        return stop_words_list

def remove_stop_words(word_list):
    return list(set(word_list).difference(stop_words_list))

def convert_to_stem_words(word_list):
    stem_words = [stemmer.stem(word) for word in word_list]

    return stem_words

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

class InvertedIndex:
    def __init__(self) -> None:
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, Any] = {}
        self.term_frequencies: Dict[int, Counter] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        # Tokenize text, then add each token to index with document ID
        # Add full text to docmap by doc_id
        self.docmap[doc_id] = text

        # tokenize
        tokens = tokenize(text)

        # Add tokens to index
        # Also build term frequencies
        term_freq = Counter()
        
        for token in tokens:
            index = self.index.get(token)
            if index is None:
                index = set()
                self.index[token] = index
            index.add(doc_id)
            term_freq[token] += 1
        self.term_frequencies[doc_id] = term_freq

    def get_documents(self, term: str) -> List[int]:
        # Set the document ID for a given token
        # return as a list sorted in asc order
        doc_id_list = self.index.get(term.lower())
        return sorted(doc_id_list)
    
    def get_tf(self, doc_id, term) -> int:
        tokens = term.split()
        if len(tokens) > 1:
            raise ValueError("Only one word allowed for the parameter 'term'")
        term_freq = self.term_frequencies[doc_id]
        return term_freq[term]

    def build(self) -> None:
        # Iterate over all movies and add them to both index and docmap
        with open('data/movies.json', 'r') as file:
            data = json.load(file)
            data = sorted(data['movies'], key=lambda k: k['id'])

            for m in data:
                concat = f"{m['title']} {m['description']}"
                self.__add_document(int(m['id']), concat)

            self.save()

    def save(self):
        # Save to disk using pickle.dump
        # Create cache folder if it doesn't exist in data/cache
        # cache/index.pkl
        # cache/docmap.pkl
        # create folder if it doesn't exist
        os.makedirs('cache', exist_ok=True)

        # Store data (serialize)
        with open('cache/index.pkl', 'wb') as handle:
            pickle.dump(self.index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('cache/docmap.pkl', 'wb') as handle:
            pickle.dump(self.docmap, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('cache/term_frequencies.pkl', 'wb') as handle:
            pickle.dump(self.term_frequencies, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        # load using picke.load
        try:
            with open('cache/index.pkl', 'rb') as handle:
                self.index = pickle.load(handle)
            with open('cache/docmap.pkl', 'rb') as handle:
                self.docmap = pickle.load(handle)
            with open('cache/term_frequencies.pkl', 'rb') as handle:
                self.term_frequencies = pickle.load(handle)
        except Exception as e:
            print(e)
    pass

def main() -> None:
    
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Generates inverted indexes for movies")

    term_freq_parser = subparsers.add_parser("tf", help="Get the term frequency in a document")
    term_freq_parser.add_argument("doc_id", type=int, help="Document id")
    term_freq_parser.add_argument("term", type=str, help="Lookup term")

    # Create an instance of InvertedIndex
    ii = InvertedIndex()

    try:
        ii.load()
        data = ii.index
    except Exception as e:
        print(e)

    # Load stop words (sets module-level `stop_words_list`)
    load_stop_words()

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print("Searching for: " + args.query)

            max_results = 5

            search_tokens = tokenize(args.query)
            print(f"Search tokens: {search_tokens}")
            result_list = []
            for token in search_tokens:
                try:
                    for doc_id in ii.get_documents(token):
                        if len(result_list) >= max_results:
                            break
                        result_list.append((doc_id, ii.docmap[doc_id]))

                    print(result_list)
                except Exception as e:
                    print(f"No results found for token '{token}'")
        case "tf":
            term = args.term
            doc_id = args.doc_id
            print(f"Term frequency for '{term}' in document {doc_id}")
            print(ii.get_tf(doc_id, term))
        case "build":
            ii.build()
        case _:
            parser.print_help()

    pass

if __name__ == "__main__":
    main()
