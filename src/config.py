import os
import torch

class Config:
    """
    Configuration class to store hyperparameters, file paths, model names, etc.
    Modify these as needed for different experiments.

    Attributes:
        MODEL_NAME (str): The base model or checkpoint name for embeddings/finetuning.
        DEVICE (str): Device to use ('cuda' if GPU is available, otherwise 'cpu').
        DATASET_NAME (str): Name of the dataset (e.g., HuggingFace hub ID).
        DATASET_CONFIG (str): Specific configuration name of the dataset.
        SPLIT_NAME (str): Which split to use by default (train, test, etc.).
        CONTEXT_COUNT (int): Number of context passages to retrieve.
        DECISIONS (list): Possible classification decisions for QA tasks.
        SAMPLE_SIZE (int): How many records to sample from the dataset (if needed).
        TEST_SIZE (int): Number of records in the test set.
        TRAIN_SIZE (int): Number of records in the training set.
        VAL_SIZE (int): Number of records in the validation set.
        NORMALIZE (bool): Whether to L2-normalize embeddings after encoding.
        TOP_K (int): How many top documents to retrieve for each query.
        EPOCHS (int): Number of training epochs.
        BATCH_SIZE (int): Batch size for training.
        LOGGING_STEPS (int): Frequency (in steps) for logging during training.
        WARMUP_STEPS (int): Number of warmup steps for learning rate scheduler.
        LEARNING_RATE (float): Base learning rate for finetuning.
        INFERENCE_BATCH_SIZE (int): Batch size used during inference/evaluation.
        RANDOM_SEED (int): Random seed for reproducibility.
        BASE_DIR (str): The root directory of your project.
        DATA_DIR (str): Path to the data folder.
        RAW_DATA_DIR (str): Path to the raw data subfolder.
        PROCESSED_DATA_DIR (str): Path to the processed data subfolder.
        TRAIN_SPLIT_PATH (str): Path to the training split file.
        TEST_SPLIT_PATH (str): Path to the test split file.
        VAL_SPLIT_PATH (str): Path to the validation split file.
        TSDAE_TRAIN_PATH (str): Path to the TSDAE train file.
        MLM_TRAIN_PATH (str): Path to the MLM train file.
        TEST_CONTEXTS_PATH (str): Path to the test contexts file.
        TEST_QUESTIONS_PATH (str): Path to the test questions file.
        EVAL_DATA_PATH (str): Path to the evaluation data file.
        MODELS_OUTPUT_DIR (str): Directory where finetuned models will be saved.
    """

    # Model / Device
    MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    DATASET_NAME: str = "qiaojin/PubMedQA"
    DATASET_CONFIG: str = "pqa_artificial"
    SPLIT_NAME: str = "train"

    # Filtering / Sampling
    CONTEXT_COUNT: int = 3
    DECISIONS = ["yes", "no"]
    SAMPLE_SIZE: int = 2000
    TEST_SIZE: int = 1000
    TRAIN_SIZE: int = 2000
    VAL_SIZE: int = 500

    # Embedding
    NORMALIZE: bool = True

    # Retrieval
    TOP_K: int = 3

    # Training Hyperparameters
    EPOCHS: int = 1
    BATCH_SIZE: int = 32
    LOGGING_STEPS: int = 100
    WARMUP_STEPS: int = 100

    # Additional Hyperparams
    LEARNING_RATE: float = 5e-5       # Common default for many transformers
    INFERENCE_BATCH_SIZE: int = 64    # Often smaller/larger than training batch
    RANDOM_SEED: int = 42            # For reproducibility

    # Base directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")

    # Data splits
    TRAIN_SPLIT_PATH: str = os.path.join(RAW_DATA_DIR, "pubmedqa_train.pkl")
    TEST_SPLIT_PATH: str = os.path.join(RAW_DATA_DIR, "pubmedqa_test.pkl")
    VAL_SPLIT_PATH: str = os.path.join(RAW_DATA_DIR, "pubmedqa_val.pkl")
    TSDAE_TRAIN_PATH: str = os.path.join(PROCESSED_DATA_DIR, "tsdae_train.pkl")
    MLM_TRAIN_PATH: str = os.path.join(PROCESSED_DATA_DIR, "mlm_train.pkl")

    TEST_CONTEXTS_PATH: str = os.path.join(PROCESSED_DATA_DIR, "test_contexts.pkl")
    TEST_QUESTIONS_PATH: str = os.path.join(PROCESSED_DATA_DIR, "test_questions.pkl")
    EVAL_DATA_PATH: str = os.path.join(PROCESSED_DATA_DIR, "eval_data.pkl")

    # Model output
    MODELS_OUTPUT_DIR: str = "models/finetuned_mnr"

    def __init__(self):
        """
        Optionally set your random seed or do any
        additional initialization tasks here.
        """
        torch.manual_seed(self.RANDOM_SEED)
        if self.DEVICE == "cuda":
            torch.cuda.manual_seed_all(self.RANDOM_SEED)

