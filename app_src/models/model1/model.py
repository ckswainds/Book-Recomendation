import os
import joblib
import scipy.sparse as sp
import pandas as pd

from logger import get_logger
from constants import *
from sklearn.feature_extraction.text import TfidfVectorizer
from exception import ModelTrainingError

from entity.artifact_entity import BuildFeaturesArifact, ModelTrainerArtifact
from entity.config_entity import ModelTrainerConfig

logger = get_logger(log_filename="model_trainer.log")


class RecommendationModelTrainer:
    """
    Trains TF-IDF models for books and papers and saves vectorizers
    and matrices for later use in prediction.

    Args:
        build_feature_artifact (BuildFeaturesArifact): paths to modified/interim CSVs.
        model_trainer_config (ModelTrainerConfig): trainer config containing artifact paths.
    """

    def __init__(self, build_feature_artifact: BuildFeaturesArifact, model_trainer_config: ModelTrainerConfig):
        try:
            self.build_feature_artifact = build_feature_artifact
            self.model_trainer_config = model_trainer_config
            logger.info("Initialized RecommendationModelTrainer")
        except Exception as e:
            logger.exception("Failed to initialize RecommendationModelTrainer: %s", e)
            raise ModelTrainingError("Initialization failed") from e

    def build_tfidf_matrix(self, series_text: pd.Series, max_features: int = 20000):
        """
        Fit a TfidfVectorizer on provided text series and return (vectorizer, matrix).
        """
        try:
            tfidf = TfidfVectorizer(stop_words="english", max_features=max_features)
            mat = tfidf.fit_transform(series_text.fillna("").astype(str))
            logger.debug("Built TF-IDF matrix with shape %s", mat.shape)
            return tfidf, mat
        except Exception as e:
            logger.exception("Error while building TF-IDF matrix: %s", e)
            raise ModelTrainingError("TF-IDF matrix building failed") from e

    def initiate_model_training(self) -> ModelTrainerArtifact:
        """
        Main entry point for training and saving TF-IDF models and matrices.

        Steps:
        - Load modified books and papers data.
        - Train TF-IDF vectorizers for each.
        - Save vectorizers and matrices.
        - Return ModelTrainerArtifact describing saved paths.
        """
        try:
            logger.info("Starting TF-IDF model training pipeline")

            # Load interim modified data
            df_books = pd.read_csv(self.build_feature_artifact.modified_books_data_filepath)
            df_papers = pd.read_csv(self.build_feature_artifact.modified_papers_data_filepath)
            logger.info("Loaded %d books and %d papers for TF-IDF training", len(df_books), len(df_papers))

            # --- Books TF-IDF ---
            if os.path.exists(self.model_trainer_config.book_tfidf_model_filepath) and os.path.exists(self.model_trainer_config.book_tfidf_matrix_filepath):
                logger.info("Book TF-IDF artifacts already exist. Skipping training.")
            else:
                logger.info("Training new Book TF-IDF model")
                os.makedirs(os.path.dirname(self.model_trainer_config.book_tfidf_model_filepath), exist_ok=True)
                os.makedirs(os.path.dirname(self.model_trainer_config.book_tfidf_matrix_filepath), exist_ok=True)

                book_tfidf_vectorizer, book_tfidf_matrix = self.build_tfidf_matrix(df_books["combined_text"], max_features=5000)
                joblib.dump(book_tfidf_vectorizer, self.model_trainer_config.book_tfidf_model_filepath)
                sp.save_npz(self.model_trainer_config.book_tfidf_matrix_filepath, book_tfidf_matrix)
                logger.info("Saved Book TF-IDF vectorizer and matrix")

            # --- Papers TF-IDF ---
            if os.path.exists(self.model_trainer_config.paper_tfidf_model_filepath) and os.path.exists(self.model_trainer_config.paper_tfidf_matrix_filepath):
                logger.info("Paper TF-IDF artifacts already exist. Skipping training.")
            else:
                logger.info("Training new Paper TF-IDF model")
                os.makedirs(os.path.dirname(self.model_trainer_config.paper_tfidf_model_filepath), exist_ok=True)
                os.makedirs(os.path.dirname(self.model_trainer_config.paper_tfidf_matrix_filepath), exist_ok=True)

                paper_tfidf_vectorizer, paper_tfidf_matrix = self.build_tfidf_matrix(df_papers["combined_text"], max_features=5000)
                joblib.dump(paper_tfidf_vectorizer, self.model_trainer_config.paper_tfidf_model_filepath)
                sp.save_npz(self.model_trainer_config.paper_tfidf_matrix_filepath, paper_tfidf_matrix)
                logger.info("Saved Paper TF-IDF vectorizer and matrix")

            # Return artifact
            artifact = ModelTrainerArtifact(
                book_tfidf_model_filepath=self.model_trainer_config.book_tfidf_model_filepath,
                paper_tfidf_model_filepath=self.model_trainer_config.paper_tfidf_model_filepath,
                book_tfidf_matrix_filepath=self.model_trainer_config.book_tfidf_matrix_filepath,
                paper_tfidf_matrix_filepath=self.model_trainer_config.paper_tfidf_matrix_filepath,

            )

            logger.info("TF-IDF model training completed successfully")
            return artifact

        except Exception as e:
            logger.exception("Error in initiate_model_training: %s", e)
            raise ModelTrainingError("TF-IDF model training failed") from e


def main():
    """
    Quick test runner for RecommendationModelTrainer.
    Trains and saves TF-IDF models and matrices for books and papers.
    """
    try:
        build_feat_artifact = BuildFeaturesArifact(
            modified_books_data_filepath="data/interim/modified_books.csv",
            modified_papers_data_filepath="data/interim/modified_papers.csv",
        )
        trainer_cfg = ModelTrainerConfig()

        trainer = RecommendationModelTrainer(build_feat_artifact, trainer_cfg)
        artifact = trainer.initiate_model_training()

        print("TF-IDF Training Completed Successfully")
        print("Artifacts:")
        print("  Book TF-IDF model:", artifact.book_tfidf_model_filepath)
        print("  Paper TF-IDF model:", artifact.paper_tfidf_model_filepath)
        print("  Book matrix:", artifact.book_tfidf_matrix_filepath)
        print("  Paper matrix:", artifact.paper_tfidf_matrix_filepath)

    except ModelTrainingError as e:
        logger.error("ModelTrainingError in main: %s", e)
        print("Training failed:", e)
    except Exception as e:
        logger.exception("Unexpected error in RecommendationModelTrainer main: %s", e)
        print("Unexpected error:", e)


if __name__ == "__main__":
    main()
