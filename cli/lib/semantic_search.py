import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

def embed_text(text):
    sm = SemanticSearch()
    embedding = sm.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    sm = SemanticSearch()
    with open('data/movies.json', 'r') as file:
        data = json.load(file)
    sm.load_or_create_embeddings(data['movies'])

    print(f"Number of docs:   {len(sm.documents)}")
    print(f"Embeddings shape: {sm.embeddings.shape[0]} vectors in {sm.embeddings.shape[1]} dimensions")

class SemanticSearch:
    def __init__(self) -> None:
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
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

    def verify_model(self) -> None:
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")
