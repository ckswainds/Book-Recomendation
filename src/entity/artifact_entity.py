from dataclasses import dataclass




@dataclass
class DataIngestionArtifact:
    is_ingestion_successful:bool
    ingested_books_data_filepath:str
    ingested_papers_data_filepath:str