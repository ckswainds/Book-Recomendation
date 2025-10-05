import os
import json
import joblib
import scipy.sparse as sp
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from logger import get_logger
from constants import *
from exception import ModelTrainingError
from entity.artifact_entity import BuildFeaturesArifact
from entity.config_entity import ModelTrainerConfig

logger = get_logger(log_filename="predict.log")


class RecommenderPredictor:
    """
    RecommenderPredictor loads trained TF-IDF models and matrices,
    computes similarity scores for a given query, and returns top-N
    recommendations as JSON for books and papers.
    """

    def __init__(self, query: str, build_feature_artifact: BuildFeaturesArifact, model_trainer_config: ModelTrainerConfig):
        self.query = query
        self.build_feature_artifact = build_feature_artifact
        self.model_trainer_config = model_trainer_config
        logger.info("Initialized RecommenderPredictor for query: %s", query)

    def _load_artifacts(self):
        """Load TF-IDF vectorizers, matrices, and datasets."""
        try:
            logger.info("Loading TF-IDF vectorizers and matrices from disk")
            # Load trained vectorizers & matrices
            book_tfidf_vectorizer = joblib.load(self.model_trainer_config.book_tfidf_model_filepath)
            paper_tfidf_vectorizer = joblib.load(self.model_trainer_config.paper_tfidf_model_filepath)

            book_tfidf_matrix = sp.load_npz(self.model_trainer_config.book_tfidf_matrix_filepath)
            paper_tfidf_matrix = sp.load_npz(self.model_trainer_config.paper_tfidf_matrix_filepath)

            # Load processed data
            df_books = pd.read_csv(self.build_feature_artifact.modified_books_data_filepath)
            df_paper = pd.read_csv(self.build_feature_artifact.modified_papers_data_filepath)

            return book_tfidf_vectorizer, book_tfidf_matrix, paper_tfidf_vectorizer, paper_tfidf_matrix, df_books, df_paper
        except Exception as e:
            logger.exception("Error loading artifacts: %s", e)
            raise ModelTrainingError("Failed to load artifacts") from e

    def _compute_similarity(self, query, tfidf_vectorizer, tfidf_matrix):
        """Compute cosine similarity for a query."""
        qv = tfidf_vectorizer.transform([query])
        sims = cosine_similarity(qv, tfidf_matrix).flatten()
        return sims

    def predict(self, top_books: int = 3, top_papers: int = 2):
        """Return top-N book and paper recommendations as JSON."""
        try:
            (
                book_tfidf_vectorizer,
                book_tfidf_matrix,
                paper_tfidf_vectorizer,
                paper_tfidf_matrix,
                df_books,
                df_paper,
            ) = self._load_artifacts()

            # Compute similarities
            book_sims = self._compute_similarity(self.query, book_tfidf_vectorizer, book_tfidf_matrix)
            paper_sims = self._compute_similarity(self.query, paper_tfidf_vectorizer, paper_tfidf_matrix)

            # Final score same as training logic
            df_books["sim_score"] = book_sims
            df_books["final_score"] = (
                0.55 * df_books.get("sim_score", 0)
                + 0.25 * df_books.get("rating_score", 0)
                + 0.15 * df_books.get("recency_score", 0)
                + 0.05 * df_books.get("page_score", 0)
            )

            df_paper["sim_score"] = paper_sims
            df_paper["final_score"] = (
                0.60 * df_paper.get("sim_score", 0)
                + 0.30 * df_paper.get("citations_score", 0)
                + 0.10 * df_paper.get("recency_score", 0)
            )

            # Select top recommendations
            top_books_df = df_books.sort_values("final_score", ascending=False).head(top_books)
            top_papers_df = df_paper.sort_values("final_score", ascending=False).head(top_papers)

            # Convert to JSON
            result = {
                "query": self.query,
                "top_books": top_books_df[["title", "authors","description","publisher","publishedDate","avgrating"]].to_dict(orient="records"),
                "top_papers": top_papers_df[["Title","Abstract","Authors","Year","Citations"]].to_dict(orient="records"),
            }

            logger.info("Prediction successful for query: %s", self.query)
            return json.dumps(result, indent=4)

        except Exception as e:
            logger.exception("Prediction failed: %s", e)
            raise ModelTrainingError("Prediction pipeline failed") from e


def main():
    """Quick test for prediction."""
    query = "deep learning for image recognition"

    build_feat_artifact = BuildFeaturesArifact(
        modified_books_data_filepath="data/interim/modified_books.csv",
        modified_papers_data_filepath="data/interim/modified_papers.csv",
    )
    trainer_cfg = ModelTrainerConfig()

    predictor = RecommenderPredictor(query, build_feat_artifact, trainer_cfg)
    output_json = predictor.predict(top_books=3, top_papers=2)

    print(output_json)


if __name__ == "__main__":
    main()
