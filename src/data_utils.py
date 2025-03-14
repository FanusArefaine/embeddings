# src/data_utils.py

import os
import pickle
import random
import sys

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from src.config import Config

sys.path.append('..')

def load_or_create_pubmedqa_splits(cfg: Config, random_seed: int = 42):
    """
    Loads the train/val/test splits from disk if they exist.
    Otherwise, creates them by:
      1) Loading the PubMedQA dataset from HuggingFace.
      2) Filtering by final_decision and context count.
      3) Shuffling and splitting according to cfg.TRAIN_SIZE, cfg.VAL_SIZE, cfg.TEST_SIZE.
      4) Saving each split to pickle files.

    Returns:
        (train_df, val_df, test_df): pandas DataFrames for each split.
    """
    # 1) If all split files exist, load them from disk
    split_paths_exist = (
        os.path.exists(cfg.TRAIN_SPLIT_PATH)
        and os.path.exists(cfg.VAL_SPLIT_PATH)
        and os.path.exists(cfg.TEST_SPLIT_PATH)
    )
    if split_paths_exist:
        print("Found existing train/val/test splits. Loading from disk...")
        train_df = pd.read_pickle(cfg.TRAIN_SPLIT_PATH)
        val_df = pd.read_pickle(cfg.VAL_SPLIT_PATH)
        test_df = pd.read_pickle(cfg.TEST_SPLIT_PATH)
        return train_df, val_df, test_df

    # 2) Otherwise, load the dataset from HuggingFace
    print("No existing splits found. Creating new splits from PubMedQA...")
    dataset = load_dataset(cfg.DATASET_NAME, cfg.DATASET_CONFIG)
    raw_df = dataset[cfg.SPLIT_NAME].to_pandas()

    # 3) Filter rows
    raw_df = raw_df[raw_df["final_decision"].isin(cfg.DECISIONS)]
    raw_df["num_contexts"] = raw_df["context"].apply(lambda x: len(x["contexts"]))
    raw_df = raw_df[raw_df["num_contexts"] == cfg.CONTEXT_COUNT].copy()
    raw_df.drop_duplicates(subset="pubid", keep="first", inplace=True)

    # 4) Shuffle
    raw_df = raw_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # 5) Check there’s enough data
    total_required = cfg.TRAIN_SIZE + cfg.VAL_SIZE + cfg.TEST_SIZE
    if len(raw_df) < total_required:
        raise ValueError(f"Not enough rows for the required splits. Need {total_required}, found {len(raw_df)}.")

    # 6) Split
    train_df = raw_df.iloc[:cfg.TRAIN_SIZE].copy()
    val_df = raw_df.iloc[cfg.TRAIN_SIZE : cfg.TRAIN_SIZE + cfg.VAL_SIZE].copy()
    test_df = raw_df.iloc[cfg.TRAIN_SIZE + cfg.VAL_SIZE : cfg.TRAIN_SIZE + cfg.VAL_SIZE + cfg.TEST_SIZE].copy()

    # 7) Save to disk
    os.makedirs(os.path.dirname(cfg.TRAIN_SPLIT_PATH), exist_ok=True)
    train_df.to_pickle(cfg.TRAIN_SPLIT_PATH)
    val_df.to_pickle(cfg.VAL_SPLIT_PATH)
    test_df.to_pickle(cfg.TEST_SPLIT_PATH)
    print("Saved new train/val/test splits to disk.")

    return train_df, val_df, test_df


def load_test_data(cfg: Config):
    """
    Loads the test DataFrame.
    If cfg.TEST_SPLIT_PATH doesn't exist, we run load_or_create_pubmedqa_splits first.
    Returns: a pandas DataFrame for the test split.
    """
    # 1) If test split file exists, load it
    if os.path.exists(cfg.TEST_SPLIT_PATH):
        print(f"Loading test split from {cfg.TEST_SPLIT_PATH} ...")
        test_df = pd.read_pickle(cfg.TEST_SPLIT_PATH)
        return test_df

    # 2) Otherwise, create the splits
    print("Test split not found. Generating splits now...")
    _, _, test_df = load_or_create_pubmedqa_splits(cfg)
    return test_df


def extract_docs_and_questions(cfg: Config, df: pd.DataFrame):
    """
    From a PubMedQA DataFrame (test or any other),
    extracts:
      1) A list of context docs, each with {doc_id, pubid, text}
      2) A list of question entries, each with {pubid, question}

    This function also checks if we have already saved them to
    cfg.TEST_CONTEXTS_PATH and cfg.TEST_QUESTIONS_PATH. If they
    exist, it loads them instead.

    Returns:
       (all_context_docs, question_entries)
         all_context_docs: list of dicts with keys ['doc_id', 'pubid', 'text']
         question_entries: list of dicts with keys ['pubid', 'question']
    """
    # 1) If already saved, just load them
    contexts_file = cfg.TEST_CONTEXTS_PATH
    questions_file = cfg.TEST_QUESTIONS_PATH

    if os.path.exists(contexts_file) and os.path.exists(questions_file):
        print("Loading cached context docs and questions from disk...")
        with open(contexts_file, 'rb') as f_ctx:
            all_context_docs = pickle.load(f_ctx)
        with open(questions_file, 'rb') as f_q:
            question_entries = pickle.load(f_q)
        return all_context_docs, question_entries

    # 2) Otherwise, extract from the provided DataFrame
    print("Extracting context docs and questions...")
    all_context_docs = []
    question_entries = []

    for _, row in df.iterrows():
        pubid = row['pubid']
        question_text = row['question']
        context_snippets = row['context']['contexts']

        # Save question
        question_entries.append({
            'pubid': pubid,
            'question': question_text
        })

        # Save each snippet as a doc
        for i, snippet in enumerate(context_snippets):
            doc_id = f"{pubid}_{i}"
            all_context_docs.append({
                'doc_id': doc_id,
                'pubid': pubid,
                'text': snippet
            })

    # 3) Cache them to disk
    os.makedirs(os.path.dirname(contexts_file), exist_ok=True)
    with open(contexts_file, 'wb') as f_ctx:
        pickle.dump(all_context_docs, f_ctx)
    with open(questions_file, 'wb') as f_q:
        pickle.dump(question_entries, f_q)

    print(f"Cached context docs -> {contexts_file}")
    print(f"Cached question entries -> {questions_file}")

    return all_context_docs, question_entries
