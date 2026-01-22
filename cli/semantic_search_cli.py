#!/usr/bin/env python3

import argparse
from pydoc import doc
import re

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

    semantic_chunk = subparsers.add_parser("semantic_chunk", help="Chunk text for processing")
    semantic_chunk.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk.add_argument("--max-chunk-size", type=int, default=4, help="Max ize of each chunk")
    semantic_chunk.add_argument("--overlap", type=int, default=0, help="Overlap between chunks")

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

        case "embed_text":
            from lib.semantic_search import embed_text
            embed_text(args.text)
        case "verify":
            from lib.semantic_search import SemanticSearch
            sm = SemanticSearch()
            sm.verify_model()
        case "verify_embeddings":
            from lib.semantic_search import verify_embeddings
            verify_embeddings()
        case "embedquery":
            from lib.semantic_search import embed_query_text
            embed_query_text(args.query)
        case "search":
            from lib.semantic_search import SemanticSearch
            from lib.semantic_search import open_json_file

            sm = SemanticSearch()
            movies_path = 'data/movies.json'
            movies_data = open_json_file(movies_path)
            sm.load_or_create_embeddings(movies_data['movies'])
            results = sm.search(args.query, args.limit)
            for score, doc in results:
                print(f"{doc['title']} (score: {score:.4f})\n  {doc['description']}\n")

        case "semantic_chunk":
            max_chunk_size = args.max_chunk_size
            overlap = args.overlap
            
            separator = " "
            sentences = re.split(r"(?<=[.!?])\s+", args.text)
            
            print(f"Semantically chunking {len(separator.join(sentences))} characters")
            
            chunk_count = 1
            index = 0
            
            while index <= len(sentences):
                print(f"{chunk_count}. {separator.join(sentences[index:index + max_chunk_size])}")
                chunk_count += 1
                index += max_chunk_size
                if (index >= len(sentences)):
                    break

                index -= overlap
                pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
