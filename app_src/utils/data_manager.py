# All required data files
import os
import logging
from dagshub import get_repo_bucket_client
import streamlit as st
import dagshub
# Configure logger
logger = logging.getLogger(__name__)

# Files to ensure exist locally
DATA_FILES = {
    "data/raw/Ml_books.csv": "data/raw/Ml_books.csv",
    "data/raw/all_papers.csv": "data/raw/all_papers.csv",
    "data/interim/modified_books.csv": "data/interim/modified_books.csv",
    "data/interim/modified_papers.csv": "data/interim/modified_papers.csv",
    "data/processed/matrices/sentence_transformer_book_matrix.npy": "data/processed/matrices/sentence_transformer_book_matrix.npy",
    "data/processed/matrices/sentence_transformer_paper_matrix.npy": "data/processed/matrices/sentence_transformer_paper_matrix.npy",
}


def get_dagshub_boto_client():
    """
    Creates and returns an authenticated boto client for the DagsHub repo.
    Works both locally and on Streamlit Cloud (if DAGSHUB_TOKEN is in secrets).
    """
    try:
        user = os.getenv("DAGSHUB_USER") or st.secrets.get("DAGSHUB_USER")
        token = os.getenv("DAGSHUB_TOKEN") or st.secrets.get("DAGSHUB_TOKEN")
        repo = "book-paper-recommender"
        
        if not user or not token:
            raise ValueError("Missing DAGSHUB_USER or DAGSHUB_TOKEN environment variables.")

        os.environ["DAGSHUB_USER"] = user
        os.environ["DAGSHUB_TOKEN"] = token
        
        # dagshub.auth.add_app_token(token)
        dagshub.auth.add_app_token(token)
        boto_client = get_repo_bucket_client(f"{user}/{repo}", flavor="boto")
        logger.info("✅ Successfully created DagsHub boto client.")
        return boto_client, user, repo
    except Exception as e:
        logger.error(f"❌ Failed to initialize DagsHub boto client: {e}", exc_info=True)
        raise


def download_data_from_remote():
    """
    Ensures that all required data files exist locally.
    Downloads any missing ones silently using DagsHub's boto client.
    """
    try:
        boto_client, user, repo = get_dagshub_boto_client()

        for remote_path, local_path in DATA_FILES.items():
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if not os.path.exists(local_path):
                try:
                    logger.info(f"📥 Downloading {remote_path} from DagsHub S3...")
                    boto_client.download_file(
                        Bucket=f"{user}/{repo}",
                        Key=remote_path,
                        Filename=local_path,
                    )
                    logger.info(f"✅ Downloaded {remote_path}")
                except Exception as e:
                    logger.error(f"⚠️ Failed to download {remote_path}: {e}", exc_info=True)
            else:
                logger.debug(f"File already exists: {local_path}")

        logger.info("✅ All data files verified and available.")
    except Exception as e:
        logger.error(f"❌ Error during data download: {e}", exc_info=True)
        raise


@st.cache_resource
def ensure_all_data_available():
    """
    Ensures all required data files are available locally.
    Runs silently (no Streamlit messages).
    """
    download_data_from_remote()
