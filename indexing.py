import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config

# Global cache for the model to prevent reloading on every call
_model = None

def create_embeddings(text_chunks):
    """Generates embeddings for a list of text chunks using local Sentence Transformers."""
    global _model
    if _model is None:
        _model = SentenceTransformer(Config.EMBEDDING_MODEL)
    embeddings = _model.encode(text_chunks)
    return np.array(embeddings).astype('float32')

def save_to_faiss(embeddings, metadata):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, "vector_db/index.faiss")