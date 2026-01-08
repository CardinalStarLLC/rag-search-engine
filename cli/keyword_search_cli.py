#!/usr/bin/env python3

import argparse
from inverted_index import InvertedIndex
from search_utils import *

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
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Limit")


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


            search_tokens = tokenize(args.query)
            print(f"Search tokens: {search_tokens}")
            result_list = []
            for token in search_tokens:
                try:
                    for doc_id in ii.get_documents(token):
                        if len(result_list) >= MAX_SEARCH_RESULTS:
                            break
                        result_list.append((doc_id, ii.docmap[doc_id]))
                        print((doc_id, ii.docmap[doc_id]))
                except Exception as e:
                    print(f"No results found for token '{token}'")
        case "bm25idf":
            bm25idf = ii.get_bm25_idf(term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25search":
            ii.bm25_search(args.query, args.limit)
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
