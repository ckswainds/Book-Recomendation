import os
import dagshub
import dagshub.auth
from dagshub import get_repo_bucket_client
from loguru import logger
import streamlit as st
# Mapping of remote → local paths
DATA_FILES = {
    "data/raw/Ml_books.csv": "data/raw/Ml_books.csv",
    "data/raw/all_papers.csv": "data/raw/all_papers.csv",
    "data/interim/modified_books.csv": "data/interim/modified_books.csv",
    "data/interim/modified_papers.csv": "data/interim/modified_papers.csv",
    "data/processed/matrices/sentence_transformer_book_matrix.npy": "data/processed/matrices/sentence_transformer_book_matrix.npy",
    "data/processed/matrices/sentence_transformer_paper_matrix.npy": "data/processed/matrices/sentence_transformer_paper_matrix.npy",
}


def authenticate_dagshub():
    """
    Authenticate DagsHub client in headless (non-OAuth) environments
    like Streamlit Cloud or GitHub Actions.
    """
    user = os.getenv("DAGSHUB_USER")
    token = os.getenv("DAGSHUB_TOKEN")

    if not user or not token:
        raise EnvironmentError("Missing DAGSHUB_USER or DAGSHUB_TOKEN environment variables.")

    # Register credentials with dagshub’s internal auth manager
    dagshub.auth.add_token(user, token)
    logger.info(f"Authenticated with DagsHub as {user}")

    return user, token


def get_boto_client(user: str, repo: str):
    """
    Initialize boto-style client for accessing DagsHub S3 storage.
    """
    try:
        boto_client = get_repo_bucket_client(f"{user}/{repo}", flavor="boto")
        logger.info("📦 DagsHub boto client initialized successfully.")
        return boto_client
    except Exception as e:
        logger.error(f"Failed to initialize DagsHub boto client: {e}", exc_info=True)
        raise


def download_data_from_dagshub():
    """
    Download all required data files from DagsHub S3 into local folders.
    Automatically skips already existing files.
    """
    logger.info("Starting data download from DagsHub...")

    user, _ = authenticate_dagshub()
    repo = "book-paper-recommender"
    boto_client = get_boto_client(user, repo)

    for remote_path, local_path in DATA_FILES.items():
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if os.path.exists(local_path):
            logger.info(f" Skipping {local_path} (already exists).")
            continue

        try:
            logger.info(f"⬇Downloading {remote_path} → {local_path}")
            boto_client.download_file(
                Bucket=repo,
                Key=remote_path,
                Filename=local_path
            )
            logger.info(f" Downloaded: {local_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to download {remote_path}: {e}")

    logger.info("Data download process completed.")



@st.cache_resource
def ensure_all_data_available():
    """
    Ensures all required data files are available locally.
    Runs silently (no Streamlit messages).
    """
    download_data_from_dagshub()
