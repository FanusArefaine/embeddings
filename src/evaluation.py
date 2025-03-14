# src/evaluation.py

import numpy as np
import math

def compute_precision_at_k(retrieved_indices, relevant_indices, k):
    """
    Computes Precision@k for a single query.
    
    Args:
        retrieved_indices (list[int]): Ranked list of doc indices retrieved.
        relevant_indices (set[int]): Set of doc indices that are actually relevant.
        k (int): The cutoff rank.

    Returns:
        float: Precision@k = (# relevant in top k) / k.
    """
    if k <= 0:
        raise ValueError("k must be >= 1")

    top_k = retrieved_indices[:k]
    num_relevant_in_top_k = len(set(top_k).intersection(relevant_indices))
    precision_at_k = num_relevant_in_top_k / k
    return precision_at_k


def compute_recall_at_k(retrieved_indices, relevant_indices, k):
    """
    Computes Recall@k for a single query.
    
    Args:
        retrieved_indices (list[int]): Ranked list of doc indices retrieved.
        relevant_indices (set[int]): Set of doc indices that are actually relevant.
        k (int): The cutoff rank.

    Returns:
        float: Recall@k = (# relevant in top k) / (total # relevant).
    """
    num_relevant = len(relevant_indices)
    if num_relevant == 0:
        return 0.0
    top_k = set(retrieved_indices[:k])
    found_relevant = len(top_k.intersection(relevant_indices))
    recall_at_k = found_relevant / num_relevant
    return recall_at_k


def compute_f1_at_k(retrieved_indices, relevant_indices, k):
    """
    Computes F1 score at a specific cutoff k, for a single query.
    F1 = 2 * (precision * recall) / (precision + recall).
    
    Args:
        retrieved_indices (list[int]): Ranked list of doc indices retrieved.
        relevant_indices (set[int]): Set of doc indices that are actually relevant.
        k (int): The cutoff rank.

    Returns:
        float: The F1 score at k. If both precision and recall are 0, returns 0.
    """
    prec = compute_precision_at_k(retrieved_indices, relevant_indices, k)
    rec = compute_recall_at_k(retrieved_indices, relevant_indices, k)
    if (prec + rec) == 0:
        return 0.0
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1


def compute_average_precision(retrieved_indices, relevant_indices):
    """
    Computes Average Precision (AP) for a single query.
    
    AP = average of precision at each rank where a relevant document is found.

    Args:
        retrieved_indices (list[int]): Ranked list of doc indices retrieved.
        relevant_indices (set[int]): Set of doc indices that are actually relevant.

    Returns:
        float: The Average Precision for this query.
    """
    num_relevant = len(relevant_indices)
    if num_relevant == 0:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(retrieved_indices, start=1):
        if doc_id in relevant_indices:
            hits += 1
            precision_at_i = hits / i
            sum_precisions += precision_at_i

    return sum_precisions / num_relevant


def compute_mean_average_precision(list_of_retrieved, list_of_relevant):
    """
    Computes Mean Average Precision (MAP) across multiple queries.
    
    Args:
        list_of_retrieved (list[list[int]]): Each element is a ranked list of doc indices for one query.
        list_of_relevant (list[set[int]]): Each element is a set of relevant doc indices for that query.
        
    Returns:
        float: The MAP for all the queries provided.

    Example Usage:
        map_score = compute_mean_average_precision(
            list_of_retrieved=[
                [10, 3, 5, 7],  # query1 docs
                [2, 1, 9, 11],  # query2 docs
            ],
            list_of_relevant=[
                {3, 7},         # query1 relevant
                {1, 9},         # query2 relevant
            ]
        )
    """
    if len(list_of_retrieved) != len(list_of_relevant):
        raise ValueError("Mismatch in number of queries between retrieved and relevant data.")

    average_precisions = []
    for retrieved, relevant in zip(list_of_retrieved, list_of_relevant):
        ap = compute_average_precision(retrieved, relevant)
        average_precisions.append(ap)

    if not average_precisions:
        return 0.0
    return float(np.mean(average_precisions))


def compute_mrr(retrieved_indices, relevant_indices):
    """
    Computes MRR (reciprocal rank) for a single query.
    
    MRR = 1 / (rank of the first relevant doc).
    If none is relevant, returns 0.

    Args:
        retrieved_indices (list[int]): Ranked list of doc indices retrieved.
        relevant_indices (set[int]): Set of doc indices that are actually relevant.

    Returns:
        float: The reciprocal rank of the first relevant doc.
    """
    for rank, doc_id in enumerate(retrieved_indices, start=1):
        if doc_id in relevant_indices:
            return 1.0 / rank
    return 0.0


def compute_mean_reciprocal_rank(list_of_retrieved, list_of_relevant):
    """
    Computes the Mean Reciprocal Rank (MRR) across multiple queries.
    
    Args:
        list_of_retrieved (list[list[int]]): Each element is a ranked list of doc indices for one query.
        list_of_relevant (list[set[int]]): Each element is a set of relevant doc indices for that query.

    Returns:
        float: The MRR across all the queries.
    """
    if len(list_of_retrieved) != len(list_of_relevant):
        raise ValueError("Mismatch in number of queries between retrieved and relevant data.")

    rr_scores = []
    for retrieved, relevant in zip(list_of_retrieved, list_of_relevant):
        rr = compute_mrr(retrieved, relevant)
        rr_scores.append(rr)

    if not rr_scores:
        return 0.0
    return float(np.mean(rr_scores))


def compute_r_precision(retrieved_indices, relevant_indices):
    """
    Computes R-Precision for a single query.
    R = number of relevant docs. We look at the top-R retrieved docs and check how many are relevant.
    
    Args:
        retrieved_indices (list[int]): Ranked list of doc indices retrieved.
        relevant_indices (set[int]): Set of doc indices that are actually relevant.

    Returns:
        float: R-Precision, i.e., (# relevant in top R) / R.
    """
    R = len(relevant_indices)
    if R == 0:
        return 0.0
    top_R = retrieved_indices[:R]
    num_relevant_in_top_R = len(set(top_R).intersection(relevant_indices))
    return num_relevant_in_top_R / R


def compute_ndcg(retrieved_indices, relevance_scores, k=None):
    """
    Computes nDCG (Normalized Discounted Cumulative Gain) at k (if k provided),
    for a single query. This supports graded relevance.
    
    Args:
        retrieved_indices (list[int]): Ranked list of doc indices retrieved.
        relevance_scores (dict[int, float]): A dict mapping doc_id -> relevance grade (0,1,2,...).
        k (int, optional): The cutoff rank. If None, uses the entire ranking.

    Returns:
        float: nDCG at k (or full rank if k is None).
    
    Notes:
        If you only have binary relevance, then relevance_scores[doc_id] should be 1 or 0.
        For more graded usage, you can have multiple levels, e.g. 0,1,2,3.
    """
    if k is None or k > len(retrieved_indices):
        k = len(retrieved_indices)

    # Compute DCG
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_indices[:k], start=1):
        rel_score = relevance_scores.get(doc_id, 0.0)
        dcg += rel_score / math.log2(rank + 1)

    # Compute IDCG (ideal DCG) by sorting doc_ids by their relevance scores, highest first
    # and then taking the top k.
    sorted_by_relevance = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
    ideal_top_k = sorted_by_relevance[:k]

    idcg = 0.0
    for rank, (_doc_id, score) in enumerate(ideal_top_k, start=1):
        idcg += score / math.log2(rank + 1)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg