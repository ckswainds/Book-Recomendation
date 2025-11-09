import os
import subprocess
from loguru import logger


def pull_dvc_data():
    """
    Pulls all DVC-tracked data from the configured remote (DagsHub S3 bucket).
    This replaces manual boto downloads.
    Works silently in background on Streamlit Cloud or locally.
    """
    dagshub_user = os.getenv("DAGSHUB_USER")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")

    if not dagshub_user or not dagshub_token:
        logger.warning(
            "DAGSHUB_USER or DAGSHUB_TOKEN not found in environment. "
            "DVC may fail to authenticate with remote."
        )

    try:
        logger.info("Starting DVC data synchronization from remote...")
        result = subprocess.run(
            ["dvc", "pull"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        logger.info("✅ DVC data successfully pulled from DagsHub remote.")
        logger.debug(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error("❌ DVC pull failed.")
        logger.error(e.stderr)
        raise
    except Exception as e:
        logger.error(f"Unexpected error while pulling data: {e}")
        raise


def download_data_from_remote():
    """
    Ensures that all data directories exist before the app starts.
    If data is missing, triggers DVC pull automatically.
    """
    required_paths = [
        "data/raw/Ml_books.csv",
        "data/raw/all_papers.csv",
        "data/interim/modified_books.csv",
        "data/interim/modified_papers.csv",
        "data/processed/matrices/sentence_transformer_book_matrix.npy",
        "data/processed/matrices/sentence_transformer_paper_matrix.npy",
    ]

    missing = [p for p in required_paths if not os.path.exists(p)]
    if missing:
        logger.warning(f"⚠️ Missing files detected: {missing}")
        pull_dvc_data()
    else:
        logger.info("✅ All data files already exist locally.")




@st.cache_resource
def ensure_all_data_available():
    """
    Ensures all required data files are available locally.
    Runs silently (no Streamlit messages).
    """
    download_data_from_remote()
