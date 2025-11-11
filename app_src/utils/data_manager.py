
import streamlit as st
import os
from dagshub import get_repo_bucket_client
from loguru import logger


DATA_FILES = {
    "data/raw/Ml_books.csv": "data/raw/Ml_books.csv",
    "data/raw/all_papers.csv": "data/raw/all_papers.csv",
    "data/interim/modified_books.csv": "data/interim/modified_books.csv",
    "data/interim/modified_papers.csv": "data/interim/modified_papers.csv",
    "data/processed/matrices/sentence_transformer_book_matrix.npy": "data/processed/matrices/sentence_transformer_book_matrix.npy",
    "data/processed/matrices/sentence_transformer_paper_matrix.npy": "data/processed/matrices/sentence_transformer_paper_matrix.npy",
}


def download_data_from_dagshub():
    """Download all required data files from DagsHub S3 into local folders."""
    user = os.getenv("DAGSHUB_USER")
    token = os.getenv("DAGSHUB_TOKEN")
    repo = "book-paper-recommender"

    if not user or not token:
        raise ValueError("Missing DAGSHUB_USER or DAGSHUB_TOKEN environment variables.")

    logger.info(f"Accessing as {user}")
    boto_client = get_repo_bucket_client(f"{user}/{repo}", flavor="boto")

    for remote_path, local_path in DATA_FILES.items():
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            logger.info(f"Downloading {remote_path}")
            boto_client.download_file(
                Bucket=repo,
                Key=remote_path,
                Filename=local_path
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to download {remote_path}: {e}")



@st.cache_resource
def ensure_all_data_available():
    """
    Ensures all required data files are available locally.
    
    """
    download_data_from_dagshub()
