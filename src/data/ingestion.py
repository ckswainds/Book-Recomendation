from constants import *
from logger import get_logger
from exception import DataLoadError,MissingDataFieldError
from entity.config_entity import DataIngestionConfig
from entity.artifact_entity import DataIngestionArtifact
import requests
import json
from typing import List, Union, Any
import time
import os
import yaml

logger = get_logger(log_filename="data_ingestion.log")

API_KEY=os.getenv(GOOGLE_BOOKS_API)
if not API_KEY:
    logger.error("Google Books API key is not set. Please set the environment variable '%s'.", GOOGLE_BOOKS_API)
    print(f"Error: Google Books API key is not set. Please set the environment variable '{GOOGLE_BOOKS_API}'.")
    exit(1)
else:
    logger.info("API KEY is set Proceed Further")


class DataIngestion:
    """
    Handles the ingestion of books and research papers data from external APIs.
    """

    def __init__(self):
        """
        Initializes the DataIngestion class and its configuration.
        """
        try:
            logger.info("Initializing DataIngestion...")
            self.data_ingestion_config = DataIngestionConfig()
            logger.info("DataIngestionConfig loaded successfully.")
        except Exception as e:
            logger.error("Error occurred in DataIngestion __init__: %s", e)
            raise

    def load_books_data(self, queries: List[str]) -> list:
        """
        Loads books data from Google Books API for the given queries.

        Args:
            queries (List[str]): List of search queries.

        Returns:
            list: List of books data.
        """
        try:
            logger.info("Loading books data for queries: %s", queries)
            all_books = []
            
            
            for q in queries:
                q = f'intitle:"{q}"'
                for start in range(0, 80,10):
                    url = f"https://www.googleapis.com/books/v1/volumes?q={q}&maxResults=40&startIndex={start}&key={API_KEY}"

                    response = requests.get(url)
                    data = response.json()
                    items = data.get("items", [])
                    all_books.extend(items)
                    logger.debug("Fetched %d items for query '%s' at startIndex %d.", len(items), q, start)
                    time.sleep(0.8)
                    
                    
            logger.info("Books data loaded successfully. Total books: %d", len(all_books))
            return all_books
        
        except DataLoadError as e:
            logger.error("Failed to load books data: %s", e)
            raise e
        
        except Exception as e:
            logger.error("Unexpected error in load_books_data: %s", e)
            raise

    def load_papers_data(self, queries: List[str], limit=100, max_results=300) -> list:
        """
        Fetches research papers from Semantic Scholar API.

        Args:
            queries (List[str]): List of search keywords.
            limit (int): Results per API call (max 100).
            max_results (int): Total number of results to fetch.

        Returns:
            list: List of research papers data.
        """
        try:
            logger.info("Loading papers data for queries: %s", queries)
            all_papers = []
            base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            fields = "title,abstract,authors,url,year,citationCount,venue"
            for query in queries:
                papers = []
                for offset in range(0, max_results, limit):
                    url = f"{base_url}?query={query}&limit={limit}&offset={offset}&fields={fields}"
                    response = requests.get(url)
                    if response.status_code != 200:
                        logger.error("Error fetching '%s': %d", query, response.status_code)
                        break
                    data = response.json()
                    items = data.get("data", [])
                    if not items:
                        logger.warning("No items returned for query '%s' at offset %d.", query, offset)
                        break
                    for item in items:
                        item["searchQuery"] = query
                    papers.extend(items)
                    logger.debug("Fetched %d papers for query '%s' at offset %d.", len(items), query, offset)
                    time.sleep(1)
                all_papers.extend(papers)
            logger.info("Papers data loaded successfully. Total papers: %d", len(all_papers))
            return all_papers
        except DataLoadError as e:
            logger.error("Failed to load papers data: %s", e)
            raise e
        except Exception as e:
            logger.error("Unexpected error in load_papers_data: %s", e)
            raise

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion process for books and papers.

        Returns:
            DataIngestionArtifact: Artifact containing ingestion status and file paths.
        """
        try:
            logger.info("Starting data ingestion process...")
            
            with open("configs/book_topics.yaml", "r") as f:
                book_topics = yaml.safe_load(f)["book_topics"]
                
            with open("configs/paper_topics.yaml", "r") as f:
                paper_topics = yaml.safe_load(f)["paper_topics"]
            
            keywords_books = [kw for topic in book_topics for kw in topic["keywords"]]
            keywords_papers = [kw for topic in paper_topics for kw in topic["keywords"]]
            
            
            all_books = self.load_books_data(keywords_books)
            all_papers = self.load_papers_data(keywords_papers)

            data_ingestion_dir = os.path.dirname(self.data_ingestion_config.ingested_books_data_filepath)
            os.makedirs(data_ingestion_dir, exist_ok=True)
            logger.info("Saving ingested books data to %s", self.data_ingestion_config.ingested_books_data_filepath)
            with open(self.data_ingestion_config.ingested_books_data_filepath, "w",encoding="utf-8") as f:
                json.dump(all_books, f, ensure_ascii=False, indent=4)

            logger.info("Saving ingested papers data to %s", self.data_ingestion_config.ingested_papers_data_filepath)
            with open(self.data_ingestion_config.ingested_papers_data_filepath, "w",encoding="utf-8") as f:
                json.dump(all_papers, f, ensure_ascii=False, indent=4)

            data_ingestion_artifact = DataIngestionArtifact(
                is_ingestion_successful=True,
                ingested_books_data_filepath=self.data_ingestion_config.ingested_books_data_filepath,
                ingested_papers_data_filepath=self.data_ingestion_config.ingested_papers_data_filepath
            )
            logger.info("Data ingestion completed successfully.")
            return data_ingestion_artifact
        except Exception as e:
            logger.error("Error during the initiate_data_ingestion: %s", e)
            raise e



def main():
    """
    Main function to test the DataIngestion process.
    """
    try:
        logger.info("Testing DataIngestion...")
        ingestion = DataIngestion()
        artifact = ingestion.initiate_data_ingestion()
        logger.info("Ingestion Artifact: %s", artifact)
        print("Ingestion successful:", artifact.is_ingestion_successful)
        print("Books data file:", artifact.ingested_books_data_filepath)
        print("Papers data file:", artifact.ingested_papers_data_filepath)
    except Exception as e:
        logger.error("Error in main: %s", e)
        print("Error during ingestion:", e)

if __name__ == "__main__":
    main()
