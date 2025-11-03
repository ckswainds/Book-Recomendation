import os
from dataclasses import dataclass
from constants import *


@dataclass
class DataIngestionConfig:
    data_ingestion_dir:str=os.path.join(FEATURE_STORE_DIRNAME,FEATURE_STORE_EXTERNAL_DATA_FOLDER)
    ingested_books_data_filepath:str=os.path.join(data_ingestion_dir,BOOKS_DATA_FILENAME)
    ingested_papers_data_filepath:str=os.path.join(data_ingestion_dir,PAPERS_DATA_FILENAME)
    

@dataclass
class DataCleaningConfig:
    cleaned_data_dir:str=os.path.join(CLEANED_DATA_DIRNAME,CLEANED_DATA_FOLDER)
    cleaned_books_data_filepath:str=os.path.join(cleaned_data_dir,CLEANED_BOOKS_DATA_FILENAME)
    cleaned_papers_data_filepath:str=os.path.join(cleaned_data_dir,CLEANED_PAPERS_DATA_FILENAME)
    
    
@dataclass
class BuildFeatureConfig:
    modified_data_dir:str=os.path.join(MODIFIED_DATA_DIRNAME,MODIFIED_DATA_FOLDER)
    modified_books_data_filepath:str=os.path.join(modified_data_dir,MODIFIED_BOOKS_DATA_FILENAME)
    modified_papers_data_filepath:str=os.path.join(modified_data_dir,MODIFIED_PAPERS_DATA_FILENAME)


@dataclass
class ModelTrainerConfig:
    """
    Configuration for model trainer outputs and related artifact filepaths.

    Creates separate sub-folders under the trainer directory:
      - matrices_dir : stores sparse matrix files (.npz)
      - objects_dir  : stores pickled/serialized objects (models, vectorizers)
      - final_dir    : stores final CSV outputs

    Values are derived from constants so changes in constants propagate here.
    """
    model_trainer_dir: str = os.path.join(MODEL_OUTPUT_DIR, MODEL_OUTPTUT_DATA_FOLDER)

    # separate sub-folders for organization
    matrices_dir: str = os.path.join(model_trainer_dir, "matrices")
    objects_dir: str = os.path.join(model_trainer_dir, "models")
    final_dir: str = os.path.join(model_trainer_dir, "final")

    # object (model/vectorizer) filepaths go to objects_dir
    book_tfidf_model_filepath: str = os.path.join(objects_dir, BOOK_TF_IDF_MODEL)
    paper_tfidf_model_filepath: str = os.path.join(objects_dir, PAPER_TF_IDF_MODEL)

    # matrix filepaths go to matrices_dir
    book_tfidf_matrix_filepath: str = os.path.join(matrices_dir, BOOK_TFIDF_MATRIX)
    paper_tfidf_matrix_filepath: str = os.path.join(matrices_dir, PAPER_TFIDF_MATRIX)
    
    
   
@dataclass
class ModelConfig:
    model_trainer_dir: str = os.path.join(MODEL_OUTPUT_DIR, MODEL_OUTPTUT_DATA_FOLDER)
    matrices_dir: str = os.path.join(model_trainer_dir, "matrices")
    sentence_transformer_model_path:str=SENTENCE_TRANSFORMER_MODEL_DIR
    sentence_transformer_book_matrix_filepath: str = os.path.join(matrices_dir, SENTENCE_TRANSFORMER_BOOK_MATRIX)
    sentence_transformer_paper_matrix_filepath: str = os.path.join(matrices_dir, SENTENCE_TRANSFORMER_PAPER_MATRIX)