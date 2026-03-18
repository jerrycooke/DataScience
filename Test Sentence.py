from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")   # very popular lightweight model
print("Model loaded successfully!")

# Example usage
sentences = ["This is a test sentence.", "Another example."]
embeddings = model.encode(sentences)
print(embeddings.shape)          # should be (2, 384)