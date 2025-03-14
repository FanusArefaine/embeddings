# src/retrieval.py

import faiss
import numpy as np

def build_faiss_index(doc_embeddings: np.ndarray):
    """
    Builds a FAISS IndexFlatIP index and places it on GPU if available.
    Assumes doc_embeddings are L2-normalized if you want cosine similarity.
    
    Returns:
        gpu_index (faiss.Index)
    """
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(doc_embeddings)
    return gpu_index


def search_top_k(gpu_index, query_embeddings, top_k=3, do_normalize=False):
    """
    For each query embedding, retrieves top_k results from the FAISS index.
    Optionally normalizes query before searching.

    Args:
        gpu_index: a FAISS index on GPU
        query_embeddings (np.ndarray): shape (num_queries, embedding_dim)
        top_k (int): how many results
        do_normalize (bool): if True, L2-normalize query embeddings

    Returns:
        indices (np.ndarray): shape (num_queries, top_k)
        distances (np.ndarray): shape (num_queries, top_k)
    """
    if do_normalize:
        faiss.normalize_L2(query_embeddings)
    distances, idxs = gpu_index.search(query_embeddings, top_k)
    return idxs, distances