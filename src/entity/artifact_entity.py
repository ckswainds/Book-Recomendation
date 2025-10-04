from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class DataIngestionArtifact:
    is_ingestion_successful: bool
    ingested_books_data_filepath: str
    ingested_papers_data_filepath: str


@dataclass
class DataCleaningArtifact:
    cleaned_books_data_filepath: str
    cleaned_papers_data_filepath: str


@dataclass
class BuildFeaturesArifact:
    modified_books_data_filepath: str
    modified_papers_data_filepath: str



@dataclass
class ModelTrainerArtifact:
    """
    Artifact produced by the model trainer stage.

    Includes locations for serialized models, sparse matrices and final CSVs,
    plus optional flags/metrics.
    """
    book_tfidf_model_filepath: str
    paper_tfidf_model_filepath: str
    book_tfidf_matrix_filepath: str
    paper_tfidf_matrix_filepath: str

    # Final CSV outputs produced by the trainer/post-processing
    books_final_filepath: str
    papers_final_filepath: str


