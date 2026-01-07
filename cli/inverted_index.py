import json
import math
import os
import pickle
from collections import Counter, defaultdict
from search_utils import *
from typing import List, Dict, Any, Set, Iterable

CACHE_DIR = 'cache'

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: Dict[int, str] = {}
        self.term_frequencies: Dict[int, Counter] = {}
        self.doc_lengths: Dict[int, int] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    # Add a document to the index
    def __add_document(self, doc_id: int, text: str) -> None:
        # Tokenize text, then add each token to index with document ID
        # Add full text to docmap by doc_id
        self.docmap[doc_id] = text

        # tokenize
        tokens = tokenize(text)

        # Store document length
        self.doc_lengths[doc_id] = len(tokens)

        # Add tokens to index
        # Also build term frequencies
        term_freq = Counter()
        
        for token in tokens:
            self.index[token].add(doc_id)
            term_freq[token] += 1
        self.term_frequencies[doc_id] = term_freq

    def __get_avg_doc_length(self) -> float:
        # Calculate the average document length across all documents
        doc_count = len(self.doc_lengths)
        if doc_count <= 0: 
            return 0.0
        doc_length_sum = 0
        for doc_id in self.doc_lengths:
            doc_length_sum += self.doc_lengths[doc_id]
        print (f"Total Document Length Sum: {doc_length_sum}, Document Count: {doc_count}")
        return round(doc_length_sum / doc_count, 2)

    # Get BM25 IDF for a given term
    def get_bm25_idf(self, term: str) -> float:
        # log((N - df + 0.5) / (df + 0.5) + 1)
        doc_count = len(self.docmap)
        term_docs = self.get_documents(term)
        df = len(term_docs)
        idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)
        return idf
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b = BM25_B):
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        tf = self.get_tf(doc_id, term)
        # Length normalization factor
        length_norm = round(1 - b + b * (doc_length / avg_doc_length), 2)
        bm25_tf = round((tf * (k1 + 1)) / (tf + k1 * length_norm), 2)
        print(f"BM25 TF: {bm25_tf}")

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
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Store data (serialize)
        with open(self.index_path, 'wb') as handle:
            pickle.dump(self.index, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved {self.index_path}")
        with open(self.docmap_path, 'wb') as handle:
            pickle.dump(self.docmap, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved {self.docmap_path}")
        with open(self.term_frequencies_path    , 'wb') as handle:
            pickle.dump(self.term_frequencies, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved {self.term_frequencies_path}")
        with open(self.doc_lengths_path, 'wb') as handle:
            pickle.dump(self.doc_lengths, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved {self.doc_lengths_path}")

    def load(self):
        # load using picke.load
        try:
            with open(self.index_path, 'rb') as handle:
                self.index = pickle.load(handle)
                print(f"Loaded {self.index_path}")
            with open(self.docmap_path, 'rb') as handle:
                self.docmap = pickle.load(handle)
                print(f"Loaded {self.docmap_path}")
            with open(self.term_frequencies_path, 'rb') as handle:
                self.term_frequencies = pickle.load(handle)
                print(f"Loaded {self.term_frequencies_path}")
            with open(self.doc_lengths_path, 'rb') as handle:
                self.doc_lengths = pickle.load(handle)
                print(f"Loaded {self.doc_lengths_path}")
        except Exception as e:
            print(e)
    pass