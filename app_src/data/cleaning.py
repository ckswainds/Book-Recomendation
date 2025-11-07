import os
import sys
import json
from app_src.constants import *
from app_src.exception import BookRecommenderError
from app_src.entity.artifact_entity import DataIngestionArtifact, DataCleaningArtifact
from app_src.entity.config_entity import DataCleaningConfig
from app_src.logger import get_logger
import pandas as pd
import yaml

logger = get_logger(log_filename="cleaning.log")

class Cleaning:
    """
    Handles cleaning and preprocessing of ingested books and papers data.
    """

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_cleaning_config: DataCleaningConfig):
        """
        Initializes the Cleaning class with ingestion artifact and cleaning config.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Artifact containing paths to ingested data.
            data_cleaning_config (DataCleaningConfig): Configuration for cleaned data output paths.
        """
        try:
            logger.info("Initializing Cleaning class...")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_cleaning_config = data_cleaning_config
            logger.info("Cleaning class initialized successfully.")
        except Exception as e:
            logger.error("Error in Cleaning initialization: %s", e)
            raise BookRecommenderError("Error in the cleaning initiation")

    def clean_and_save_papers(self) -> None:
        """
        Cleans the ingested papers data and saves the refined data as CSV.
        """
        try:
            logger.info("Loading papers data from %s", self.data_ingestion_artifact.ingested_papers_data_filepath)
         
            with open(self.data_ingestion_artifact.ingested_papers_data_filepath, "r", encoding="utf-8") as f:
                papers = json.load(f)

            refined = []
            for p in papers:
                refined.append({
                    "SearchQuery": p.get("searchQuery", ""),
                    "Title": p.get("title", ""),
                    "Abstract": p.get("abstract", ""),
                    "Authors": ", ".join([a.get("name", "") for a in p.get("authors", [])]),
                    "Year": p.get("year", ""),
                    "Citations": p.get("citationCount", 0),
                    "Venue": p.get("venue", ""),
                    "URL": p.get("url", "")
                })

            df_papers = pd.DataFrame(refined)
            df_papers = df_papers.drop_duplicates(subset="Title", keep="first")

            paper_dir = os.path.dirname(self.data_cleaning_config.cleaned_papers_data_filepath)
            os.makedirs(paper_dir, exist_ok=True)

            logger.info("Saving cleaned papers data to %s", self.data_cleaning_config.cleaned_papers_data_filepath)
            df_papers.to_csv(self.data_cleaning_config.cleaned_papers_data_filepath, index=False)
            logger.info("Cleaned papers data saved successfully.")
        except Exception as e:
            logger.error("Error cleaning papers data: %s", e)
            raise e

    def clean_and_save_books(self) -> None:
        """
        Cleans the ingested books data and saves the refined data as CSV.
        """
        try:
            logger.info("Loading books data from %s", self.data_ingestion_artifact.ingested_books_data_filepath)
            with open(self.data_ingestion_artifact.ingested_books_data_filepath, "r", encoding="utf-8") as f:
                all_books = json.load(f)
            
            with open("configs/book_topics.yaml", "r") as f:
                book_topics = yaml.safe_load(f)["book_topics"]
            
            ml_keywords = [kw for topic in book_topics for kw in topic["keywords"]]

            filtered_books = []
            for item in all_books:
                title = item["volumeInfo"].get("title", "").lower()
                desc = item["volumeInfo"].get("description", "").lower()
                if any(k in title or k in desc for k in ml_keywords):
                    filtered_books.append(item)

            books_list = []
            for item in filtered_books:
                info = item["volumeInfo"]
                books_list.append({
                    "title": info.get("title"),
                    "authors": ", ".join(info.get("authors", [])),
                    "description": info.get("description", ""),
                    "categories": ", ".join(info.get("categories", [])),
                    "publisher": info.get('publisher', []),
                    "publishedDate": info.get("publishedDate", ""),
                    "avgrating": info.get("averageRating", 0),
                    "pagecount": info.get("pageCount", 0),
                    "previewLink": info.get("previewLink", "")
                })
            df_books = pd.DataFrame(books_list)
            df_books = df_books.drop_duplicates(subset=["title"], keep="first")
            df_books = df_books[df_books["pagecount"] > 0]

            books_dir = os.path.dirname(self.data_cleaning_config.cleaned_books_data_filepath)
            os.makedirs(books_dir, exist_ok=True)

            logger.info("Saving cleaned books data to %s", self.data_cleaning_config.cleaned_books_data_filepath)
            df_books.to_csv(self.data_cleaning_config.cleaned_books_data_filepath, index=False)
            logger.info("Cleaned books data saved successfully.")
        except Exception as e:
            logger.error("Error cleaning books data: %s", e)
            raise e

    def initiate_data_cleaning(self) -> DataCleaningArtifact:
        """
        Initiates the data cleaning process for books and papers.

        Returns:
            DataCleaningArtifact: Artifact containing paths to cleaned data files.
        """
        try:
            logger.info("Starting data cleaning process...")
            self.clean_and_save_papers()
            self.clean_and_save_books()
            logger.info("Data cleaning process completed successfully.")
            return DataCleaningArtifact(
                cleaned_books_data_filepath=self.data_cleaning_config.cleaned_books_data_filepath,
                cleaned_papers_data_filepath=self.data_cleaning_config.cleaned_papers_data_filepath
            )
        except Exception as e:
            logger.error("Error during data cleaning initiation: %s", e)
            raise BookRecommenderError(e)

def main():
    """
    Main function to test the Cleaning module.
    """
    try:
        logger.info("Testing Cleaning module...")
       
        ingestion_artifact = DataIngestionArtifact(
            is_ingestion_successful=True,
            ingested_books_data_filepath="data/external/Ml_books.json",
            ingested_papers_data_filepath="data/external/all_papers.json"
        )
        cleaning_config = DataCleaningConfig()
        cleaner = Cleaning(ingestion_artifact, cleaning_config)
        artifact = cleaner.initiate_data_cleaning()
        logger.info("Cleaning Artifact: %s", artifact)
        print("Cleaning successful!")
        print("Cleaned books file:", artifact.cleaned_books_data_filepath)
        print("Cleaned papers file:", artifact.cleaned_papers_data_filepath)
    except Exception as e:
        logger.error("Error in main: %s", e)
        print("Error during cleaning:", e)

if __name__ == "__main__":
    main()