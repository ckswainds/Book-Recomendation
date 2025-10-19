import os
GOOGLE_BOOKS_API="Google_api"



#Data ingestion Constants
FEATURE_STORE_DIRNAME:str="data"
FEATURE_STORE_EXTERNAL_DATA_FOLDER:str="external"
BOOKS_DATA_FILENAME:str="Ml_books.json"
PAPERS_DATA_FILENAME:str="all_papers.json"


#Data cleaning constants
CLEANED_DATA_DIRNAME:str="data"
CLEANED_DATA_FOLDER:str="raw"
CLEANED_BOOKS_DATA_FILENAME:str="Ml_books.csv"
CLEANED_PAPERS_DATA_FILENAME:str="all_papers.csv"



#FeatureBuilding constants
MODIFIED_DATA_DIRNAME:str="data"
MODIFIED_DATA_FOLDER:str="interim"
MODIFIED_BOOKS_DATA_FILENAME:str="modified_books.csv"
MODIFIED_PAPERS_DATA_FILENAME:str="modified_papers.csv"



#Model trainer constants
MODEL_OUTPUT_DIR:str="data"
MODEL_OUTPTUT_DATA_FOLDER:str="processed"
BOOK_TF_IDF_MODEL:str="book_tfidf_vectorizer.pkl"
PAPER_TF_IDF_MODEL:str="paper_tfidf_vectorizer.pkl"
BOOK_TFIDF_MATRIX:str="book_tfidf_matrix.npz"
PAPER_TFIDF_MATRIX:str="paper_tfidf_matrix.npz"


#Model constants
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SENTENCE_TRANSFORMER_MODEL_DIR = os.path.join(ROOT_DIR , "models")
SENTENCE_TRANSFORMER_BOOK_MATRIX:str="sentence_transformer_book_matrix.npy"
SENTENCE_TRANSFORMER_PAPER_MATRIX:str="sentence_transformer_paper_matrix.npy"

APP_HOST = "0.0.0.0"
APP_PORT = 5000