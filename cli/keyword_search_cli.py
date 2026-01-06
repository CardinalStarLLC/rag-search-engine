#!/usr/bin/env python3

import argparse
import json
import math
import string
from nltk.stem import PorterStemmer
# imports for InvertedIndex
from typing import List, Dict, Any, Set, Iterable
from collections import Counter, defaultdict
import pickle
import os

BM25_K1 = 1.5
stemmer = PorterStemmer()

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

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: Dict[int, Any] = {}
        self.term_frequencies: Dict[int, Counter] = {}

    # Add a document to the index
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
            self.index[token].add(doc_id)
            term_freq[token] += 1
        self.term_frequencies[doc_id] = term_freq

    # Get BM25 IDF for a given term
    def get_bm25_idf(self, term: str) -> float:
        # log((N - df + 0.5) / (df + 0.5) + 1)
        doc_count = len(self.docmap)
        term_docs = self.get_documents(term)
        df = len(term_docs)
        idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)
        return idf
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1):
        tf = self.get_tf(doc_id, term)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1)
        return bm25_tf

    # Get list of document IDs for a given term
    def get_documents(self, term: str) -> List[int]:
        # Set the document ID for a given token
        # return as a list sorted in asc order
        doc_id_list = self.index.get(term.lower())
        if doc_id_list is None:
            return []
        return sorted(doc_id_list)
    
    # Get term frequency for a given document ID and term
    def get_tf(self, doc_id, term) -> int:
        tokens = term.split()
        if len(tokens) > 1:
            raise ValueError("Only one word allowed for the parameter 'term'")
        term_freq = self.term_frequencies[doc_id]
        return term_freq[term]
    
    # Get inverse document frequency for a given term
    def get_idf(self, term) -> float:
        documents = self.get_documents(term)
        idf = math.log((len(self.docmap) + 1) / (len(documents) + 1))
        return idf

    def build(self) -> None:
        # Iterate over all movies and add them to both index and docmap
        with open('data/movies.json', 'r') as file:
            data = json.load(file)
            #data = sorted(data['movies'], key=lambda k: k['id'])

            for m in data['movies']:
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

    inverse_document_frequency_parser = subparsers.add_parser("idf", help="Get the inverse document frequency for a term")
    inverse_document_frequency_parser.add_argument("term", type=str, help="Lookup term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get the TF-IDF for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document id")
    tfidf_parser.add_argument("term", type=str, help="Lookup term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")

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
    
    try:
        term = args.term.lower()
        term = stemmer.stem(term)
    except:
        # do nothing
        pass

    try:
        doc_id = args.doc_id
    except:
        # do nothing
        pass

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
                        print((doc_id, ii.docmap[doc_id]))
                except Exception as e:
                    print(f"No results found for token '{token}'")
        case "bm25idf":
            bm25idf = ii.get_bm25_idf(term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = ii.get_bm25_tf(doc_id, term, args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "idf":
            idf = ii.get_idf(term)
            print(f"Inverse document frequency of '{term}': {idf:.2f}")
        case "tf":
            print(f"Term frequency for '{term}' in document {doc_id}")
            print(ii.get_tf(doc_id, term))
        case "tfidf":
            tf = ii.get_tf(doc_id, term)
            print(f"Term frequency for '{term}' in document {doc_id}")
            idf = ii.get_idf(term)
            print(f"Inverse document frequency of '{term}': {idf:.2f}")
            tf_idf = tf * idf
            print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")
        case "build":
            ii.build()
        case _:
            parser.print_help()

    pass

if __name__ == "__main__":
    main()
