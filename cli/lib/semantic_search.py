import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def embed_text(text):
    sm = SemanticSearch()
    embedding = sm.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_query_text(query):
    sm = SemanticSearch()
    embedding = sm.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def verify_embeddings():
    from lib.search_utils import open_json_file
    from lib.chunked_semantic_search import ChunkedSemanticSearch

    sm = SemanticSearch()
    data = open_json_file('data/movies.json')
    sm.load_or_create_embeddings(data['movies'])

    print(f"Number of docs:   {len(sm.documents)}")
    print(f"Embeddings shape: {sm.embeddings.shape[0]} vectors in {sm.embeddings.shape[1]} dimensions")

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2') -> None:
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def build_embeddings(self, documents):
        self.documents = documents
        movies = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            movies.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(movies, show_progress_bar=True)
        with open('cache/movie_embeddings.npy', 'wb') as file:
            np.save(file, self.embeddings)
        
        return self.embeddings

    def generate_embedding(self, text):
        if text is None or text.strip() == "":
            raise ValueError("Input text must be a non-empty string.")
        
        sentences = [text]
        embedding = self.model.encode(sentences)
        return embedding[0]
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc

        if os.path.isfile('cache/movie_embeddings.npy'):
            with open('cache/movie_embeddings.npy', 'rb') as file:
                self.embeddings = np.load(file)

        if (self.embeddings is None or len(self.embeddings) != len(documents)):
            self.build_embeddings(documents)
        else:
            return self.embeddings
        
    def search(self, query, limit):
        if self.embeddings is None or self.documents is None:
            raise ValueError("Embeddings and documents must be loaded before searching.")
        
        query_embedding = self.generate_embedding(query)
        similarities = []

        for idx, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, self.documents[idx]))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:limit] 

    def verify_model(self) -> None:
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")
