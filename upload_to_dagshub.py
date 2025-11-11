from dagshub import get_repo_bucket_client
import os

DAGSHUB_USER = "Chandankumar2309"
REPO_NAME = "book-paper-recommender"


DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

boto_client = get_repo_bucket_client(f"{DAGSHUB_USER}/{REPO_NAME}", flavor="boto")

files_to_upload = {
    "data/raw/Ml_books.csv": "data/raw/Ml_books.csv",
    "data/raw/all_papers.csv": "data/raw/all_papers.csv",
    "data/interim/modified_books.csv": "data/interim/modified_books.csv",
    "data/interim/modified_papers.csv": "data/interim/modified_papers.csv",
    "data/processed/matrices/sentence_transformer_book_matrix.npy": "data/processed/matrices/sentence_transformer_book_matrix.npy",
    "data/processed/matrices/sentence_transformer_paper_matrix.npy": "data/processed/matrices/sentence_transformer_paper_matrix.npy",
}

for local, remote in files_to_upload.items():
    print(f"Uploading {local} → {remote}")
    boto_client.upload_file(
        Filename=local,
        Bucket=REPO_NAME,
        Key=remote
    )

print("All files uploaded to DagsHub S3 bucket.")
