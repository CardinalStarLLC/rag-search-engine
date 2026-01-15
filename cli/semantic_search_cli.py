#!/usr/bin/env python3

import argparse
from lib.semantic_search import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the semantic search model")

    subparsers.add_parser("verify_embeddings", help="Verify the semantic search embeddings")
    
    embed = subparsers.add_parser("embed_text", help="Verify the semantic search model")
    embed.add_argument("text", type=str, help="Text to embed")

    args = parser.parse_args()

    match args.command:
        case "embed_text":
            embed_text(args.text)
        case "verify":
            sm = SemanticSearch()
            sm.verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
