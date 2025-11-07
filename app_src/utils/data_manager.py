import os
import requests
import logging
import streamlit as st

# Configure logger
logger = logging.getLogger(__name__)

# All required data files
DATA_FILES = {
    "data/raw/Ml_books.csv": "data/raw/Ml_books.csv",
    "data/raw/all_papers.csv": "data/raw/all_papers.csv",
    "data/interim/modified_books.csv": "data/interim/modified_books.csv",
    "data/interim/modified_papers.csv": "data/interim/modified_papers.csv",
    "data/processed/matrices/sentence_transformer_book_matrix.npy": "data/processed/matrices/sentence_transformer_book_matrix.npy",
    "data/processed/matrices/sentence_transformer_paper_matrix.npy": "data/processed/matrices/sentence_transformer_paper_matrix.npy",
}


def _download_file(remote_path: str, local_path: str):
    """
    Internal function: Downloads a single file from DagsHub using token authentication.
    """
    try:
        user = "Chandankumar2309"
        repo = "book-paper-recommender"
        token = st.secrets["DAGSHUB_TOKEN"]

        url = f"https://dagshub.com/{user}/{repo}/raw/main/{remote_path}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if not os.path.exists(local_path):
            logger.info(f"Downloading {remote_path} ...")
            response = requests.get(url, auth=(user, token))
            response.raise_for_status()  
            with open(local_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Downloaded: {remote_path}")
        else:
            logger.debug(f"File already exists: {local_path}")
    except Exception as e:
        logger.error(f"Failed to download {remote_path}: {e}", exc_info=True)
        raise

@st.cache_resource
def ensure_all_data_available():
    """
    Ensures all required data files are available locally.
    Runs silently (no Streamlit messages).
    """
    for remote_path, local_path in DATA_FILES.items():
        _download_file(remote_path, local_path)
