import numpy as np
from sentence_transformers import SentenceTransformer

def embed_text(text):
    sm = SemanticSearch()
    embedding = sm.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

class SemanticSearch:
    def __init__(self) -> None:
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self, text):
        if text is None or text.strip() == "":
            raise ValueError("Input text must be a non-empty string.")
        
        sentences = [text]
        embedding = self.model.encode(sentences)
        return embedding[0]

    def verify_model(self) -> None:
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")
