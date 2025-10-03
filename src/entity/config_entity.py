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