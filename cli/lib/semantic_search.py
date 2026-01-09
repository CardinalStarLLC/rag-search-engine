from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self) -> None:
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def verify_model(self) -> None:
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")

    #sm.model.encode(text)