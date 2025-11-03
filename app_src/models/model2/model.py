import pandas as pd
# from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from entity.config_entity import ModelConfig
from entity.artifact_entity import ModelArtifact,BuildFeaturesArifact
import numpy as np
import json
import os
from logger import get_logger 
import numpy as np
from helper import get_query_embedding
# Initialize logger
logger = get_logger(__name__)

class RecommendationModel:
    """
    A recommendation model class that uses Sentence Transformers for
    content-based recommendations of books and papers.

    It trains by encoding text data into embeddings and saves them as sparse matrices.
    It recommends items based on query similarity and a weighted final score.
    """
    def __init__(self,model_config:ModelConfig,build_feature_artifact:BuildFeaturesArifact):
        """
        Initializes the RecommendationModel with configuration and artifact paths.

        Args:
            model_config (ModelConfig): Configuration object containing model paths.
            build_feature_artifact (BuildFeaturesArifact): Artifact object containing 
                                                           paths to feature-engineered dataframes.
        """
        try:
            logger.info("Initializing RecommendationModel...")
            self.model_config=model_config
            self.build_feature_artifact=build_feature_artifact
            logger.info("RecommendationModel initialized successfully.")
        except Exception as e:
            logger.error(f"Error during RecommendationModel initialization: {e}")
            raise
    
    def train(self,book_df:pd.DataFrame,paper_df:pd.DataFrame):
        """
        Trains the model by generating and saving sentence embeddings for books and papers.

        It uses a pre-trained Sentence Transformer model to encode the 'combined_text'
        column of the input dataframes and saves the normalized embeddings as
        sparse NPZ matrices.

        Args:
            book_df (pd.DataFrame): DataFrame containing book data with a 'combined_text' column.
            paper_df (pd.DataFrame): DataFrame containing paper data with a 'combined_text' column.
        """
        try:
            logger.info("Starting model training (embedding generation)...")
            # Load the Sentence Transformer model
            # model=SentenceTransformer(self.model_config.sentence_transformer_model_path)
            # logger.info(f"Loaded SentenceTransformer model from: {self.model_config.sentence_transformer_model_path}")

            # Generate embeddings
            # embeddings_books = model.encode(book_df["combined_text"].tolist(),normalize_embeddings=True)
            # embeddings_paper = model.encode(paper_df["combined_text"].tolist(),normalize_embeddings=True)
            embeddings_books=get_query_embedding(book_df["combined_text"].tolist())
            embeddings_paper=get_query_embedding(paper_df["combined_text"].tolist())
            
            logger.info(f"Books Embedding shape: {embeddings_books.shape}")
            print(f"Books Embedding shape:{embeddings_books.shape}")
            logger.info(f"Papers Embedding shape: {embeddings_paper.shape}")
            print(f"Papers Embedding shape:{embeddings_paper.shape}")

            # Create directory for matrices if it doesn't exist
            matrix_dir=os.path.dirname(self.model_config.sentence_transformer_book_matrix_filepath)
            os.makedirs(matrix_dir,exist_ok=True)
            logger.info(f"Created directory for saving matrices: {matrix_dir}")

            # Save embeddings as sparse matrices (though the output of encode is dense, sp.save_npz handles it)
            np.save(self.model_config.sentence_transformer_book_matrix_filepath, embeddings_books)
            logger.info(f"Saved book embeddings to: {self.model_config.sentence_transformer_book_matrix_filepath}")
            
            np.save(self.model_config.sentence_transformer_paper_matrix_filepath, embeddings_paper)
            logger.info(f"Saved paper embeddings to: {self.model_config.sentence_transformer_paper_matrix_filepath}")
            
            logger.info("Model training (embedding generation) completed successfully.")
        except Exception as e:
            logger.error(f"Error during training (embedding generation): {e}")
            raise
        

    
    def recommend(self,query:str,n_books:int,n_papers:int):
        """
        Generates recommendations for books and papers based on a query.

        Recommendations are based on a weighted final score that combines
        query similarity (cosine similarity with embeddings) and other features
        like rating, recency, and citations.

        Args:
            query (str): The search query text.
            n_books (int): The number of top book recommendations to return.
            n_papers (int): The number of top paper recommendations to return.

        Returns:
            str: A JSON string containing the query and the top book and paper recommendations.
        """
        try:
            logger.info(f"Starting recommendation for query: '{query}' with n_books={n_books}, n_papers={n_papers}")
            
            # Load dataframes
            df_books = pd.read_csv(self.build_feature_artifact.modified_books_data_filepath)
            logger.info(f"Loaded books data from: {self.build_feature_artifact.modified_books_data_filepath}")
            
            df_paper = pd.read_csv(self.build_feature_artifact.modified_papers_data_filepath)
            logger.info(f"Loaded papers data from: {self.build_feature_artifact.modified_papers_data_filepath}")
            
            # Load the sentence transformer model
            # model=SentenceTransformer(self.model_config.sentence_transformer_model_path)
            # logger.info(f"Loaded SentenceTransformer model for query encoding.")
            
            # Encode the query
            
            query_embedding=get_query_embedding(query)
            logger.info("Encoded query into embedding.")
            
            # Load the sentence_transformer_book_matrix
            book_matrix = np.load(self.model_config.sentence_transformer_book_matrix_filepath)
            logger.info(f"Loaded book embedding matrix from: {self.model_config.sentence_transformer_book_matrix_filepath}")
            
            # Load the sentence_transformer_paper_matrix
            paper_matrix = np.load(self.model_config.sentence_transformer_paper_matrix_filepath)
            logger.info(f"Loaded paper embedding matrix from: {self.model_config.sentence_transformer_paper_matrix_filepath}")
            
            # Calculate the similarity score for books (Cosine Similarity)
            book_sims=cosine_similarity(query_embedding.reshape(1, -1), book_matrix)
            logger.debug("Calculated cosine similarity for books.")
            
            # Calculate the similarity score for papers (Cosine Similarity)
            paper_sims=cosine_similarity(query_embedding.reshape(1, -1), paper_matrix)
            logger.debug("Calculated cosine similarity for papers.")
            
            # Reshape both the matrices to be 1D arrays
            book_sims=book_sims.reshape(-1)
            paper_sims=paper_sims.reshape(-1)
            
            # Calculate the final score for books
            df_books["sim_score"]=book_sims
            df_books["final_score"] = (
                0.55 * df_books["sim_score"] +
                0.25 * df_books["rating_score"] +
                0.15 * df_books["recency_score"] +
                0.05 * df_books["page_score"]
            )
            logger.debug("Calculated final weighted scores for books.")
            
            # Final paper scores
            df_paper["sim_score"]=paper_sims
            df_paper["final_score"] = (
                0.60 * df_paper["sim_score"] +
                0.30 * df_paper["citations_score"] +
                0.10 * df_paper["recency_score"]
            )
            logger.debug("Calculated final weighted scores for papers.")
            
            # Return the top books and papers by sorting
            top_books_df = df_books.sort_values("final_score", ascending=False).head(n_books)
            top_papers_df = df_paper.sort_values("final_score", ascending=False).head(n_papers)
            
            logger.info(f"Retrieved top {n_books} books and top {n_papers} papers.")

            # Prepare the final result dictionary
            result = {
                "query": query, 
                "top_books": top_books_df[["title", "authors","description","publisher","publishedDate","avgrating","previewLink"]].to_dict(orient="records"),
                "top_papers": top_papers_df[["Title","Authors","Year","Citations","URL"]].to_dict(orient="records"),
            }
            
            print("Prediction successful for query: %s" % query) 
            logger.info(f"Recommendation successful for query: '{query}'")
            return json.dumps(result, indent=4)
            
        except Exception as e:
            logger.error(f"Error during recommendation: {e}")
            raise