import pandas as pd
import json
from app_src.entity.config_entity import ModelConfig
from app_src.entity.artifact_entity import BuildFeaturesArifact
from app_src.models.model2.model import RecommendationModel
from app_src.logger import get_logger

# Initialize logger for the prediction script
logger = get_logger(__name__)

def start_prediction(query: str, n_books: int = 5, n_papers: int = 5) -> dict:
    """
    Generates book and research paper recommendations for a given query.

    It initializes the model with configuration and feature artifacts, 
    calls the recommendation logic, and prints the top results.

    Args:
        query (str): The search query text used to find relevant items.
        n_books (int, optional): The number of top book recommendations to return. Defaults to 5.
        n_papers (int, optional): The number of top paper recommendations to return. Defaults to 5.

    Returns:
        dict: A dictionary containing the query and the top recommended books and papers.
        
    Raises:
        Exception: If any step during prediction fails.
    """
    try:
        logger.info(f"Starting prediction for query: '{query}' ({n_books} books, {n_papers} papers).")

        # Initialize configuration and feature artifacts (paths must match where training placed the data)
        model_config = ModelConfig()
        build_feature_artifact = BuildFeaturesArifact(
        modified_books_data_filepath="data/interim/modified_books.csv",
        modified_papers_data_filepath="data/interim/modified_papers.csv"
                     )

        logger.info("Configuration and feature artifact paths loaded for prediction.")

        # Initialize the recommendation model
        model = RecommendationModel(
            model_config=model_config,
            build_feature_artifact=build_feature_artifact
        )
        logger.info("RecommendationModel instantiated for prediction.")

        # Run the recommendation process
        result_json = model.recommend(query=query, n_books=n_books, n_papers=n_papers)
        logger.info("Recommendation logic execution completed successfully.")

        result = json.loads(result_json)

        # Print the results for immediate feedback
        print("Prediction completed successfully!")
        print("\nTop Recommended Books:")
        for book in result["top_books"]:
            print(f"- {book['title']} by {book['authors']}")

        print("\nTop Recommended Research Papers:")
        for paper in result["top_papers"]:
            print(f"- {paper['Title']} by {paper['Authors']} ({paper['Year']})")

        return result

    except Exception as e:
        logger.error(f"Error occurred during prediction for query '{query}': {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Example usage for prediction
    query = "machine learning in healthcare"
    try:
        logger.info("Starting script execution (Prediction Example).")
        start_prediction(query=query, n_books=5, n_papers=5)
        logger.info(f"Prediction execution successful for query: '{query}'.")
    except Exception:
        logger.critical("Prediction script execution failed to complete.")
