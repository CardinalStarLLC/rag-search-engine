import json
import numpy as np
import os
from typing import List
from lib.semantic_search import SemanticSearch
from lib.search_utils import (
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    load_movies,
    open_json_file,
    semantic_chunk,
)


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def __cosine_similarity_score__(self, query_embedding, chunk_embedding):
        dot_product = np.dot(query_embedding, chunk_embedding)
        magnitude_query = np.linalg.norm(query_embedding)
        magnitude_chunk = np.linalg.norm(chunk_embedding)

        if (magnitude_chunk == 0 or magnitude_query == 0):
            return 0

        return dot_product / (magnitude_query * magnitude_chunk)

    def __populate_docs_and_doc_map__(self, documents):
        self.documents = documents
        movies = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            movies.append(f"{doc['title']}: {doc['description']}")

    def build_chunk_embeddings(self, documents):
        self.__populate_docs_and_doc_map__(documents)

        chunks: List[str] = []
        chunk_metadata = []

        chunk_idx = 0
        for doc_index, doc in enumerate(documents):
            if doc is None:
                continue
            sentence_chunks = semantic_chunk(doc['description'], DEFAULT_SEMANTIC_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)

            for chunk in sentence_chunks:
                chunks.append(chunk)
                data = { 'movie_idx': doc_index, 'chunk_idx': chunk_idx, 'total_chunks': len(sentence_chunks) }
                chunk_metadata.append(data)
                print(f"index: {chunk_idx} chunk: {chunk}")

                chunk_idx += 1

        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True, convert_to_tensor=True, device='cuda', batch_size=256)
        self.chunk_metadata = chunk_metadata

        with open('cache/chunk_embeddings.npy', 'wb') as file:
            np.save(file, self.chunk_embeddings)
        
        with open("cache/chunk_metadata.json", "w", encoding="utf-8") as file:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(chunks)}, file, indent=2)

        return self.chunk_embeddings
        
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.__populate_docs_and_doc_map__(documents)

        if os.path.isfile('cache/chunk_embeddings.npy'):
            with open('cache/chunk_embeddings.npy', 'rb') as file:
                self.chunk_embeddings = np.load(file)

        if os.path.isfile('cache/chunk_metadata.json'):
            self.chunk_metadata = open_json_file('cache/chunk_metadata.json')

        if ((self.chunk_embeddings is None or len(self.chunk_embeddings) <= 0)
            or (self.chunk_metadata is None or len(self.chunk_metadata) <= 0)):
            return self.build_chunk_embeddings(documents)
        else:
            return self.chunk_embeddings
        
    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)
        # print(query_embedding)
        chunk_score = [dict]

        movie_data = load_movies()
        chunk_embeddings = self.load_or_create_chunk_embeddings(movie_data)
        # print(f"Documents {self.documents}")
        # print(f"Movie metadata {self.chunk_metadata}")

        for i, chunk_embedding in enumerate(chunk_embeddings):
            score = self.__cosine_similarity_score__(chunk_embedding, query_embedding)
            chunk_score.append({ "chunk_idx": i, "movie_idx": "", "score": score})
            # print(f"chunk_embedding {chunk_embedding}")
            # print(f"chunk_idx: {i}, movie_idx: {0}, score: {score}")
            pass