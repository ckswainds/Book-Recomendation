import pandas as pd
from entity.config_entity import ModelConfig
from entity.artifact_entity import BuildFeaturesArifact, ModelArtifact
from models.model2.model import RecommendationModel
from logger import get_logger

# Initialize logger for the training script
logger = get_logger(__name__)

def start_training() -> ModelArtifact:
    """
    Orchestrates the training process for the RecommendationModel.

    This function initializes configuration, loads the preprocessed data,
    instantiates the RecommendationModel, trains it (generating and saving
    embeddings), and creates the final ModelArtifact containing the paths
    to the generated assets.

    Returns:
        ModelArtifact: An artifact object containing the file paths of the 
                       generated model matrices and the Sentence Transformer model path.
    
    Raises:
        Exception: If any step during configuration, data loading, or training fails.
    """
    try:
        logger.info("Starting recommendation model training pipeline.")
        
        # Initialize configuration and artifacts
        model_config = ModelConfig()
        logger.info("ModelConfig initialized successfully.")
        
        # NOTE: Using raw strings (r"...") to handle Windows file paths correctly.
        build_feature_artifact = BuildFeaturesArifact(
            modified_books_data_filepath=r"C:\Vscode\git\mlops\Book-Recomendation\data\interim\modified_books.csv",
            modified_papers_data_filepath=r"C:\Vscode\git\mlops\Book-Recomendation\data\interim\modified_papers.csv"
        )
        logger.info("BuildFeaturesArifact initialized with data paths.")

        # Load preprocessed data
        logger.info("Loading processed book and paper data...")
        book_df = pd.read_csv(build_feature_artifact.modified_books_data_filepath)
        paper_df = pd.read_csv(build_feature_artifact.modified_papers_data_filepath)
        logger.info(f"Loaded {len(book_df)} books and {len(paper_df)} papers.")

        # Initialize and train the model
        logger.info("Instantiating RecommendationModel and starting training...")
        model = RecommendationModel(
            model_config=model_config,
            build_feature_artifact=build_feature_artifact
        )
        model.train(book_df, paper_df)
        logger.info("Model training (embedding generation) completed successfully.")

        # Create model artifact after training
        model_artifact = ModelArtifact(
            sentence_transformer_book_matrix_filepath=model_config.sentence_transformer_book_matrix_filepath,
            sentence_transformer_paper_matrix_filepath=model_config.sentence_transformer_paper_matrix_filepath,
            sentence_transformer_model_path=model_config.sentence_transformer_model_path
        )
        logger.info("ModelArtifact created, training pipeline finished successfully.")

        return model_artifact

    except Exception as e:
        logger.error(f"Error occurred during training: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        artifact = start_training()
        logger.info(f"Pipeline execution successful. Generated artifact: {artifact}")
    except Exception:
        # The exception is re-raised and logged within start_training, 
        # but this block catches any final exceptions if they bubble up.
        logger.critical("Training pipeline failed to complete execution.")
