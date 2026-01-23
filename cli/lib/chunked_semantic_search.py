import json
import os
from typing import List

from lib.semantic_search import SemanticSearch
import numpy as np

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def __populate_docs_and_doc_map__(self, documents):
        self.documents = documents
        movies = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            movies.append(f"{doc['title']}: {doc['description']}")

    def build_chunk_embeddings(self, documents):
        from lib.search_utils import semantic_chunk

        self.__populate_docs_and_doc_map__(documents)

        chunks: List[str] = []
        chunk_metadata = []

        for doc in documents:
            if doc is None:
                continue
            sentence_chunks = semantic_chunk(doc['description'], 4, 1)

            index = 0
            for chunk in sentence_chunks:
                chunks.append(chunk)
                data = { 'movie_idx': doc['id'], 'chunk_idx': index, 'total_chunks': len(sentence_chunks) }
                chunk_metadata.append(data)

                index += 1

        self.chunk_embeddings = self.model.encode(chunks)
        self.chunk_metadata = chunk_metadata

        with open('cache/chunk_embeddings.npy', 'wb') as file:
            np.save(file, self.chunk_embeddings)
        
        with open("cache/chunk_metadata.json", "w", encoding="utf-8") as file:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(chunks)}, file, indent=2)

        return self.chunk_embeddings
        
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        from lib.search_utils import open_json_file
        self.__populate_docs_and_doc_map__(documents)

        if os.path.isfile('cache/chunk_embeddings.npy'):
            with open('cache/chunk_embeddings.npy', 'rb') as file:
                print("Loading chunk_embeddings")
                self.chunk_embeddings = np.load(file)

        if os.path.isfile('cache/chunk_metadata.json'):
            print("Loading chunk_metadata")
            self.chunk_metadata = open_json_file('cache/chunk_metadata.json')

        if ((self.chunk_embeddings is None or len(self.chunk_embeddings) <= 0)
            or (self.chunk_metadata is None or len(self.chunk_metadata) <= 0)):
            return self.build_chunk_embeddings(documents)
        else:
            return self.chunk_embeddings