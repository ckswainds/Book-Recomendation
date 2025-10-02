import os
from dataclasses import dataclass
from constants import *



# @dataclass
# class RecommendationArtifactConfig:
#     artifact_dir:str="artifacts"
    


# artifact_config=RecommendationArtifactConfig()
# #Data ingestion config
@dataclass
class DataIngestionConfig:
    data_ingestion_dir:str=os.path.join(FEATURE_STORE_DIRNAME,FEATURE_STORE_EXTERNAL_DATA_FOLDER)
    ingested_books_data_filepath:str=os.path.join(data_ingestion_dir,BOOKS_DATA_FILENAME)
    ingested_papers_data_filepath:str=os.path.join(data_ingestion_dir,PAPERS_DATA_FILENAME)
    

