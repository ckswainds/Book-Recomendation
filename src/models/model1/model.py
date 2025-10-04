import os
import sys
import joblib
import scipy.sparse as sp
import pandas as pd

from logger import get_logger
from constants import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from exception import ModelTrainingError

from entity.artifact_entity import BuildFeaturesArifact, ModelTrainerArtifact
from entity.config_entity import ModelTrainerConfig

logger = get_logger(log_filename="model_trainer.log")


class RecommendationModel:
    """
    RecommendationModel builds TF-IDF representations for books and papers,
    computes content similarity against a provided query and writes final
    scored CSV outputs. Artifacts (models, matrices, final CSVs) are saved
    under paths specified by ModelTrainerConfig.

    Args:
        query (str): User query / text to compute recommendations for.
        build_feature_artifact (BuildFeaturesArifact): artifact with paths to modified/interim CSVs.
        model_trainer_config (ModelTrainerConfig): trainer config containing artifact paths.
    """

    def __init__(self, query: str, build_feature_artifact: BuildFeaturesArifact, model_trainer_config: ModelTrainerConfig):
        try:
            self.query = query
            self.build_feature_artifact = build_feature_artifact
            self.model_trainer_config = model_trainer_config
            logger.info("Initialized RecommendationModel for query: %s", query)
        except Exception as e:
            logger.exception("Failed to initialize RecommendationModel: %s", e)
            raise ModelTrainingError("Initialization failed") from e

    def build_tfidf_matrix(self, series_text: pd.Series, max_features: int = 20000):
        """
        Fit a TfidfVectorizer on provided text series and return (vectorizer, matrix).

        Args:
            series_text: pandas Series of strings to fit TF-IDF on.
            max_features: maximum number of features for TF-IDF.

        Returns:
            (TfidfVectorizer, sparse matrix)
        """
        tfidf = TfidfVectorizer(stop_words="english", max_features=max_features)
        mat = tfidf.fit_transform(series_text.fillna("").astype(str))
        logger.debug("Built TF-IDF matrix with shape %s", mat.shape)
        return tfidf, mat

    def content_similarity_tfidf(self, query: str, tfidf: TfidfVectorizer, mat):
        """
        Compute cosine similarity between query vector and TF-IDF matrix.

        Args:
            query: query string
            tfidf: fitted TfidfVectorizer
            mat: TF-IDF matrix for corpus

        Returns:
            1D numpy array of similarity scores
        """
        qv = tfidf.transform([query])
        sims = cosine_similarity(qv, mat).flatten()
        logger.debug("Computed similarity vector of length %d", sims.shape[0])
        return sims

    def initiate_recomendation_model(self) -> ModelTrainerArtifact:
        """
        Main entrypoint for training/loading TF-IDF artifacts and producing
        final scored CSVs for books and papers.

        Steps:
        - Load interim modified_books / modified_papers CSVs.
        - Train TF-IDF vectorizers and persist vectorizer + sparse matrices if not present.
        - Compute similarity scores and aggregate into final_score fields.
        - Save final CSVs and return a ModelTrainerArtifact describing saved file paths.
        """
        try:
            logger.info("Starting model training / recommendation pipeline")
            # Load interim modified data
            df_books = pd.read_csv(self.build_feature_artifact.modified_books_data_filepath)
            df_paper = pd.read_csv(self.build_feature_artifact.modified_papers_data_filepath)
            logger.info("Loaded %d books and %d papers for training", len(df_books), len(df_paper))

            # --- Books TF-IDF artifacts ---
            if os.path.exists(self.model_trainer_config.book_tfidf_model_filepath) and os.path.exists(self.model_trainer_config.book_tfidf_matrix_filepath):
                logger.info("Loading existing book TF-IDF vectorizer and matrix from disk")
                book_tfidf_vectorizer = joblib.load(self.model_trainer_config.book_tfidf_model_filepath)
                book_tfidf_matrix = sp.load_npz(self.model_trainer_config.book_tfidf_matrix_filepath)
            else:
                # ensure directories exist
                os.makedirs(os.path.dirname(self.model_trainer_config.book_tfidf_model_filepath), exist_ok=True)
                os.makedirs(os.path.dirname(self.model_trainer_config.book_tfidf_matrix_filepath), exist_ok=True)

                logger.info("Training new book TF-IDF vectorizer on combined_text")
                book_tfidf_vectorizer, book_tfidf_matrix = self.build_tfidf_matrix(df_books["combined_text"], max_features=5000)

                # Persist artifacts
                joblib.dump(book_tfidf_vectorizer, self.model_trainer_config.book_tfidf_model_filepath)
                sp.save_npz(self.model_trainer_config.book_tfidf_matrix_filepath, book_tfidf_matrix)
                logger.info("Saved book TF-IDF vectorizer to %s and matrix to %s",
                            self.model_trainer_config.book_tfidf_model_filepath,
                            self.model_trainer_config.book_tfidf_matrix_filepath)

            # --- Papers TF-IDF artifacts ---
            if os.path.exists(self.model_trainer_config.paper_tfidf_model_filepath) and os.path.exists(self.model_trainer_config.paper_tfidf_matrix_filepath):
                logger.info("Loading existing paper TF-IDF vectorizer and matrix from disk")
                paper_tfidf_vectorizer = joblib.load(self.model_trainer_config.paper_tfidf_model_filepath)
                paper_tfidf_matrix = sp.load_npz(self.model_trainer_config.paper_tfidf_matrix_filepath)
            else:
                os.makedirs(os.path.dirname(self.model_trainer_config.paper_tfidf_model_filepath), exist_ok=True)
                os.makedirs(os.path.dirname(self.model_trainer_config.paper_tfidf_matrix_filepath), exist_ok=True)

                logger.info("Training new paper TF-IDF vectorizer on combined_text")
                paper_tfidf_vectorizer, paper_tfidf_matrix = self.build_tfidf_matrix(df_paper["combined_text"], max_features=5000)

                # Persist artifacts
                joblib.dump(paper_tfidf_vectorizer, self.model_trainer_config.paper_tfidf_model_filepath)
                sp.save_npz(self.model_trainer_config.paper_tfidf_matrix_filepath, paper_tfidf_matrix)
                logger.info("Saved paper TF-IDF vectorizer to %s and matrix to %s",
                            self.model_trainer_config.paper_tfidf_model_filepath,
                            self.model_trainer_config.paper_tfidf_matrix_filepath)

            # Compute similarity scores for query
            book_sims = self.content_similarity_tfidf(self.query, book_tfidf_vectorizer, book_tfidf_matrix)
            paper_sims = self.content_similarity_tfidf(self.query, paper_tfidf_vectorizer, paper_tfidf_matrix)

            # Attach sim + compute final score using weighted combination of features
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

            # Ensure output directories exist and save final CSVs
            os.makedirs(os.path.dirname(self.model_trainer_config.books_final_filepath), exist_ok=True)
            os.makedirs(os.path.dirname(self.model_trainer_config.papers_final_filepath), exist_ok=True)

            df_books.to_csv(self.model_trainer_config.books_final_filepath, index=False)
            df_paper.to_csv(self.model_trainer_config.papers_final_filepath, index=False)
            logger.info("Saved final books CSV to %s and papers CSV to %s",
                        self.model_trainer_config.books_final_filepath,
                        self.model_trainer_config.papers_final_filepath)

            # Return artifact describing produced files
            artifact = ModelTrainerArtifact(
                book_tfidf_model_filepath=self.model_trainer_config.book_tfidf_model_filepath,
                paper_tfidf_model_filepath=self.model_trainer_config.paper_tfidf_model_filepath,
                book_tfidf_matrix_filepath=self.model_trainer_config.book_tfidf_matrix_filepath,
                paper_tfidf_matrix_filepath=self.model_trainer_config.paper_tfidf_matrix_filepath,
                books_final_filepath=self.model_trainer_config.books_final_filepath,
                papers_final_filepath=self.model_trainer_config.papers_final_filepath,
            )
            logger.info("Model training / recommendation pipeline completed successfully")
            return artifact

        except Exception as e:
            logger.exception("Error in initiate_recomendation_model: %s", e)
            raise ModelTrainingError("Recommendation model pipeline failed") from e


def main():
    """
    Quick test runner for the RecommendationModel module.
    Instantiates the model with a sample query and runs the pipeline,
    then prints paths of created artifacts and top-5 book recommendations.
    """
    try:
        logger.info("Running RecommendationModel main test")
        # sample inputs - adjust paths if your project layout differs
        build_feat_artifact = BuildFeaturesArifact(
            modified_books_data_filepath="data/interim/modified_books.csv",
            modified_papers_data_filepath="data/interim/modified_papers.csv",
        )
        trainer_cfg = ModelTrainerConfig()

        # Example query for testing
        query = "machine learning neural networks"

        recommender = RecommendationModel(query, build_feat_artifact, trainer_cfg)
        artifact = recommender.initiate_recomendation_model()

        logger.info("Produced ModelTrainerArtifact: %s", getattr(artifact, "__dict__", str(artifact)))
        print("Artifacts produced:")
        print("  Book TF-IDF model:", artifact.book_tfidf_model_filepath)
        print("  Paper TF-IDF model:", artifact.paper_tfidf_model_filepath)
        print("  Books final CSV:", artifact.books_final_filepath)
        print("  Papers final CSV:", artifact.papers_final_filepath)

        # Display top-5 book recommendations (if file exists)
        if os.path.exists(artifact.books_final_filepath):
            df = pd.read_csv(artifact.books_final_filepath)
            top5 = df.sort_values("final_score", ascending=False).head(5)[["title", "final_score"]]
            print("\nTop 5 books:")
            print(top5.to_string(index=False))
    except ModelTrainingError as e:
        logger.error("ModelTrainingError in main: %s", e)
        print("Model training failed:", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error in RecommendationModel main: %s", e)
        print("Unexpected error:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

