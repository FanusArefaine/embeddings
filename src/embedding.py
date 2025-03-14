# src/embedding.py

import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments, Trainer
from sentence_transformers import SentenceTransformer # Import SentenceTransformer
from sentence_transformers import SentenceTransformer, models
import faiss  # for normalization if needed

def build_embedder(cfg):
    """
    Returns a SentenceTransformer model based on cfg.MODEL_NAME, on cfg.DEVICE
    """
    return SentenceTransformer(cfg.MODEL_NAME, device=cfg.DEVICE)





def embed_texts(embedder, texts, do_normalize=False, model_type=None):
    """
    Embeds a list of texts (questions or context snippets) into vectors.
    Optionally normalizes them (L2) for cosine similarity.

    Args:
        embedder: a SentenceTransformer (or similar) model
        texts: list of strings
        do_normalize: bool, whether to apply faiss.normalize_L2
    
    Returns:
        np.ndarray of shape (len(texts), embedding_dim)
    """
    embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype('float32')
    if do_normalize:
        faiss.normalize_L2(embeddings)
    return embeddings
