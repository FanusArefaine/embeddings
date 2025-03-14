import numpy as np
from src.config import Config
from src.data_utils import load_test_data, extract_docs_and_questions
from src.embedding import build_embedder, embed_texts
from src.retrieval import build_faiss_index, search_top_k
from src.metrics import evaluate_all_metrics
from src.visualization import print_metrics_table, plot_metrics_bar_chart

def run_evaluation(cfg: Config):
    """
    Runs an end-to-end evaluation (without fine-tuning) using your chosen model + data:
      1) Load test split
      2) Extract docs & questions
      3) Embed docs, build FAISS index
      4) For each question, retrieve top-k
      5) Evaluate all metrics, then visualize
    """
    # 1) Load test data
    test_df = load_test_data(cfg)
    print(f"Test size: {len(test_df)}")

    # 2) Get doc snippets & questions
    all_context_docs, questions_data = extract_docs_and_questions(cfg, test_df)
    print(f"Number of questions: {len(questions_data)}")
    print(f"Number of context docs: {len(all_context_docs)}")

    # 3) Build embedder + embed doc texts
    embedder = build_embedder(cfg)
    doc_texts = [doc['text'] for doc in all_context_docs]
    doc_embeddings = embed_texts(embedder, doc_texts, do_normalize=cfg.NORMALIZE)

    # Build FAISS index
    faiss_index = build_faiss_index(doc_embeddings)
    print(f"FAISS index size: {faiss_index.ntotal}")

    # Map index -> pubid
    pubid_map = [doc['pubid'] for doc in all_context_docs]

    # 4) Retrieve for each question, store results
    list_of_retrieved = []
    list_of_relevant = []

    for q in questions_data:
        question_text = q['question']
        question_pubid = q['pubid']

        # Embed question
        q_emb = embed_texts(embedder, [question_text], do_normalize=cfg.NORMALIZE)

        # Search top-k
        idxs, _ = search_top_k(faiss_index, q_emb, top_k=cfg.TOP_K, do_normalize=False)
        retrieved_indices = idxs[0].tolist()

        # Identify relevant doc indices
        # relevant = all docs whose pubid matches question_pubid
        relevant_indices = {i for i, pid in enumerate(pubid_map) if pid == question_pubid}

        # Collect
        list_of_retrieved.append(retrieved_indices)
        list_of_relevant.append(relevant_indices)

    # 5) Evaluate all metrics
    metrics_dict = evaluate_all_metrics(list_of_retrieved, list_of_relevant, k=cfg.TOP_K)

    # Print or visualize results
    print("\n=== Evaluation Results ===")
    print_metrics_table(metrics_dict)
    plot_metrics_bar_chart(metrics_dict)
