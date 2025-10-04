"""
Simple, professional prediction module.

Input: query string
Output: JSON-serializable dict with top 3 books and top 2 papers:
  {"top_books": [...3 items...], "top_papers": [...2 items...]}

This module loads TF-IDF vectorizers and matrices saved by the trainer,
loads the final CSVs with metadata, computes cosine similarity and returns
the requested top-K results.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.metrics.pairwise import cosine_similarity

from logger import get_logger
from entity.config_entity import ModelTrainerConfig
from exception import ModelTrainingError

logger = get_logger(log_filename="predict.log")


class SimplePredictor:
    """
    Lightweight predictor for returning top-N recommendations.

    Args:
        cfg (ModelTrainerConfig | None): optional config; defaults to ModelTrainerConfig()
    """

    def __init__(self, cfg: ModelTrainerConfig | None = None):
        self.cfg = cfg if cfg is not None else ModelTrainerConfig()
        self.book_vec = None
        self.book_mat = None
        self.paper_vec = None
        self.paper_mat = None
        logger.info("SimplePredictor initialized with config: %s", getattr(self.cfg, "__dict__", str(self.cfg)))

    def _ensure_artifacts(self):
        """Load vectorizers and matrices from disk; raise ModelTrainingError if missing."""
        # books
        if self.book_vec is None or self.book_mat is None:
            if not (os.path.exists(self.cfg.book_tfidf_model_filepath) and os.path.exists(self.cfg.book_tfidf_matrix_filepath)):
                msg = "Book TF-IDF artifacts missing"
                logger.error("%s: %s, %s", msg, self.cfg.book_tfidf_model_filepath, self.cfg.book_tfidf_matrix_filepath)
                raise ModelTrainingError(msg)
            logger.debug("Loading book TF-IDF artifacts")
            self.book_vec = joblib.load(self.cfg.book_tfidf_model_filepath)
            self.book_mat = sp.load_npz(self.cfg.book_tfidf_matrix_filepath)

        # papers
        if self.paper_vec is None or self.paper_mat is None:
            if not (os.path.exists(self.cfg.paper_tfidf_model_filepath) and os.path.exists(self.cfg.paper_tfidf_matrix_filepath)):
                msg = "Paper TF-IDF artifacts missing"
                logger.error("%s: %s, %s", msg, self.cfg.paper_tfidf_model_filepath, self.cfg.paper_tfidf_matrix_filepath)
                raise ModelTrainingError(msg)
            logger.debug("Loading paper TF-IDF artifacts")
            self.paper_vec = joblib.load(self.cfg.paper_tfidf_model_filepath)
            self.paper_mat = sp.load_npz(self.cfg.paper_tfidf_matrix_filepath)

    def _load_metadata(self):
        """Load final CSVs with metadata; raise ModelTrainingError if missing."""
        if not os.path.exists(self.cfg.books_final_filepath) or not os.path.exists(self.cfg.papers_final_filepath):
            msg = "Final CSVs missing"
            logger.error("%s: %s, %s", msg, self.cfg.books_final_filepath, self.cfg.papers_final_filepath)
            raise ModelTrainingError(msg)
        logger.debug("Loading final metadata CSVs")
        df_books = pd.read_csv(self.cfg.books_final_filepath)
        df_papers = pd.read_csv(self.cfg.papers_final_filepath)
        return df_books, df_papers

    @staticmethod
    def _top_k_from_sims(sims: np.ndarray, df: pd.DataFrame, k: int):
        """Return top-k items as list of small dicts (safe for JSON)."""
        if sims is None or sims.size == 0:
            return []
        idx = np.argsort(-sims)[:k]
        results = []
        for i in idx:
            if i < 0 or i >= len(df):
                continue
            row = df.iloc[int(i)]
            results.append({
                "title": str(row.get("title", "") or row.get("Title", "")),
                "authors": str(row.get("authors", "") or row.get("Authors", "")),
                "year": str(row.get("publishedDate", "") or row.get("Year", "")),
                "sim_score": float(sims[int(i)]) if not np.isnan(sims[int(i)]) else None,
                "final_score": float(row.get("final_score")) if ("final_score" in row and not pd.isna(row.get("final_score"))) else None,
                "url": str(row.get("previewLink", "") or row.get("URL", "")),
            })
        return results

    def predict(self, query: str, top_books: int = 3, top_papers: int = 2) -> dict:
        """
        Predict top books and papers for the given query.

        Args:
            query: user query
            top_books: number of top books to return (default 3)
            top_papers: number of top papers to return (default 2)

        Returns:
            dict with keys "top_books" and "top_papers" (lists of dicts)
        """
        try:
            logger.info("Predict called with query: %s", query)
            self._ensure_artifacts()
            df_books, df_papers = self._load_metadata()

            # compute similarities
            book_qv = self.book_vec.transform([query])
            paper_qv = self.paper_vec.transform([query])
            book_sims = cosine_similarity(book_qv, self.book_mat).flatten()
            paper_sims = cosine_similarity(paper_qv, self.paper_mat).flatten()
            logger.debug("Computed similarity vectors (books=%d, papers=%d)", book_sims.size, paper_sims.size)

            top_books_list = self._top_k_from_sims(book_sims, df_books, top_books)
            top_papers_list = self._top_k_from_sims(paper_sims, df_papers, top_papers)

            result = {"top_books": top_books_list, "top_papers": top_papers_list}
            logger.info("Prediction finished, returning %d books and %d papers", len(top_books_list), len(top_papers_list))
            return result
        except ModelTrainingError:
            raise
        except Exception as e:
            logger.exception("Prediction failed: %s", e)
            raise ModelTrainingError("Prediction pipeline failed") from e


def main():
    """
    CLI test: python predict.py "your query here"
    Prints JSON with top 3 books and top 2 papers.
    """
    try:
        query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "machine learning neural networks"
        predictor = SimplePredictor()
        output = predictor.predict(query, top_books=3, top_papers=2)
        print(json.dumps(output, indent=2))
    except ModelTrainingError as e:
        logger.error("ModelTrainingError in predict main: %s", e)
        print("Prediction failed:", e)
        sys.exit(2)
    except Exception as e:
        logger.exception("Unexpected error in predict main: %s", e)
        print("Unexpected error:", e)
        sys.exit(3)


if __name__ == "__main__":
    main()
