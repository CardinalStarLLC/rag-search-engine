#!/usr/bin/env python3

import argparse
from lib.chunked_semantic_search import ChunkedSemanticSearch
from lib.search_utils import (
    load_movies,
    semantic_chunk,
    split_text_to_sentences,
)
from lib.semantic_search import (
    SemanticSearch,
    embed_query_text,
    embed_text,
    verify_embeddings,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the semantic search model")

    subparsers.add_parser("verify_embeddings", help="Verify the semantic search embeddings")
    
    embed = subparsers.add_parser("embed_text", help="Verify the semantic search model")
    embed.add_argument("text", type=str, help="Text to embed")

    embed_query = subparsers.add_parser("embedquery", help="Verify query information")
    embed_query.add_argument("query", type=str, help="Query text")

    search = subparsers.add_parser("search", help="Search for similar documents")   
    search.add_argument("query", type=str, help="Query text")
    search.add_argument("--limit", type=int, default=5, help="Number of results to return")

    chunk = subparsers.add_parser("chunk", help="Chunk text for processing")
    chunk.add_argument("text", type=str, help="Text to chunk")
    chunk.add_argument("--chunk-size", type=int, default=200, help="Size of each chunk")
    chunk.add_argument("--overlap", type=int, default=0, help="Overlap between chunks")

    semantic_chunk_option = subparsers.add_parser("semantic_chunk", help="Chunk text for processing")
    semantic_chunk_option.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_option.add_argument("--max-chunk-size", type=int, default=4, help="Max ize of each chunk")
    semantic_chunk_option.add_argument("--overlap", type=int, default=0, help="Overlap between chunks")
    
    embed_chunks = subparsers.add_parser("embed_chunks", help="Get Chunk embeddings from text")

    search_chunked = subparsers.add_parser("search_chunked", help="Search and score a query within the embedding chunks")
    search_chunked.add_argument("query", type=str, help="Query to search documents")
    search_chunked.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")

    args = parser.parse_args()

    match args.command:
        case "chunk":
            separator = " "
            tokens = args.text.split()
            print(f"Chunking {len(separator.join(tokens))} characters")

            chunk_size = args.chunk_size
            overlap = args.overlap

            chunk_count = 1
            for chunk in range(chunk_size, len(tokens), chunk_size):
                if chunk_count <= 1:
                    print(f"{chunk_count}. {separator.join(tokens[chunk - chunk_size:chunk])}")
                else:
                    print(f"{chunk_count}. {separator.join(tokens[((chunk_count - 1) * chunk_size) - overlap:chunk])}")
                chunk_count += 1

            print(f"{chunk_count}. {separator.join(tokens[((chunk_count - 1) * chunk_size) - overlap:len(tokens)])}")

        case "embed_chunks":
            movies_data = load_movies()

            css = ChunkedSemanticSearch()
            embeddings = css.load_or_create_chunk_embeddings(movies_data)
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "embed_text":
            embed_text(args.text)
        case "verify":
            sm = SemanticSearch()
            sm.verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            sm = SemanticSearch()
            movies_data = load_movies()
            sm.load_or_create_embeddings(movies_data)
            results = sm.search(args.query, args.limit)
            for score, doc in results:
                print(f"{doc['title']} (score: {score:.4f})\n  {doc['description']}\n")
        case "search_chunked":
            css = ChunkedSemanticSearch()
            movies_data = load_movies()
            css.load_or_create_chunk_embeddings(movies_data)
            results = css.search_chunks(args.query, args.limit)

            for i, result in enumerate(results):
                score = result['score']
                print(f"\n{i+1}. {result['title']} (score: {score:.4f})")
                print(f"   {result['document']}...")
        case "semantic_chunk":
            chunk_list = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            sentences = split_text_to_sentences(args.text)
            print(f"Semantically chunking {len(" ".join(sentences))} characters")

            chunk_count = 1
            for sentences in chunk_list:
                print(f"{chunk_count}. {sentences}")
                chunk_count += 1
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
