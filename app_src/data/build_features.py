"""
Feature building / feature engineering module.

This module reads cleaned CSVs for books and papers, engineers numeric/text
features required by downstream models, and writes modified CSVs.

Changes made:
- Added module docstring, more informative logging, comments and robust checks.
- Made clean_text a @staticmethod with docstring and logging.
- Added warnings when expected columns are missing and handled them gracefully.
- Improved exception logging and re-raising with context.
"""
import os
import sys
import re
import pandas as pd
from app_src.constants import *
from app_src.logger import get_logger

from app_src.entity.artifact_entity import BuildFeaturesArifact, DataCleaningArtifact
from app_src.entity.config_entity import BuildFeatureConfig

from sklearn.preprocessing import MinMaxScaler

logger = get_logger(log_filename="build_features.log")

class BuildFeatures:
    """
    Feature engineering for books and papers.

    This class reads cleaned CSVs produced by the cleaning stage, generates
    numerical features (recency, rating, pagecounts, citations), scales them
    to [0,1] and writes modified CSVs for downstream modeling.
    """

    def __init__(self, build_features_config: BuildFeatureConfig, data_cleaning_artifact: DataCleaningArtifact):
        """
        Initialize BuildFeatures.

        Args:
            build_features_config (BuildFeatureConfig): Configuration containing output file paths.
            data_cleaning_artifact (DataCleaningArtifact): Artifact with cleaned data file paths.
        """
        try:
            # store config and artifact for use in methods
            self.build_features_config = build_features_config
            self.data_cleaning_artifact = data_cleaning_artifact
            logger.info(
                "BuildFeatures initialized with config: %s and artifact: %s",
                getattr(self.build_features_config, "__dict__", str(self.build_features_config)),
                getattr(self.data_cleaning_artifact, "__dict__", str(self.data_cleaning_artifact)),
            )
        except Exception as e:
            logger.exception("Failed to initialize BuildFeatures: %s", e)
            raise

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Basic text cleaning for combined text field.

        - Lowercases text
        - Removes punctuation (keeps word characters and whitespace)
        - Converts NaNs to empty string

        This is intentionally lightweight; downstream tokenization/vectorization
        can be applied later.

        Args:
            text: input string to clean

        Returns:
            cleaned text string
        """
        try:
            if pd.isna(text):
                return ""
            text = str(text).lower()
            # remove punctuation, keep unicode word characters and whitespace
            text = re.sub(r"[^\w\s]", "", text)
            # collapse multiple whitespace
            text = re.sub(r"\s+", " ", text).strip()
            return text
        except Exception as e:
            # Log the original input for debugging and return empty string to avoid breaking pipeline
            logger.exception("clean_text failed for input: %r ; error: %s", text, e)
            return ""

    def build_book_features(self) -> None:
        """
        Build features for books and save modified CSV.

        Steps:
        - Load cleaned books CSV.
        - Normalize/parse publishedDate and extract year.
        - Compute recency, rating and page count scores scaled to [0,1].
        - Create combined_text (title + description + categories + authors) and clean it.
        - Persist modified CSV to configured path.
        """
        try:
            logger.info("Building book features from: %s", self.data_cleaning_artifact.cleaned_books_data_filepath)
            df_books = pd.read_csv(self.data_cleaning_artifact.cleaned_books_data_filepath)

            # Ensure expected columns exist; create empty columns and warn if missing.
            expected_cols = ["publishedDate", "avgrating", "pagecount", "title", "description", "categories", "authors"]
            for c in expected_cols:
                if c not in df_books.columns:
                    logger.warning("Expected column '%s' missing from cleaned books CSV. Creating empty column.", c)
                    df_books[c] = ""

            # Extract a consistent date string then parse year
            df_books["Date_Extracted"] = df_books["publishedDate"].astype(str).str.extract(r"(\d{4}-\d{2}-\d{2}|\d{4})", expand=False)
            df_books["publishedDate"] = pd.to_datetime(df_books["Date_Extracted"], errors="coerce")
            df_books.drop(columns=["Date_Extracted"], inplace=True, errors="ignore")
            df_books["year"] = pd.to_datetime(df_books["publishedDate"], errors="coerce").dt.year

            scaler = MinMaxScaler()

            # Recency score: scale available years only
            if df_books["year"].notna().any():
                valid_years = df_books.loc[df_books["year"].notna(), "year"].astype(float).values.reshape(-1, 1)
                df_books.loc[df_books["year"].notna(), "recency_score"] = scaler.fit_transform(valid_years)
                logger.debug("Recency scores computed for %d records", valid_years.shape[0])
            else:
                df_books["recency_score"] = 0.0
                logger.warning("No valid publication years found; recency_score set to 0 for all rows")

            # Rating score
            try:
                ratings = df_books["avgrating"].fillna(0).astype(float).values.reshape(-1, 1)
                df_books["rating_score"] = MinMaxScaler().fit_transform(ratings)
            except Exception as e:
                logger.exception("Failed to compute rating_score: %s", e)
                df_books["rating_score"] = 0.0
                logger.warning("Failed to compute rating_score; set to 0 for all rows")

            # Page count score
            try:
                pages = df_books["pagecount"].fillna(0).astype(float).values.reshape(-1, 1)
                df_books["page_score"] = MinMaxScaler().fit_transform(pages)
            except Exception as e:
                logger.exception("Failed to compute page_score: %s", e)
                df_books["page_score"] = 0.0
                logger.warning("Failed to compute page_score; set to 0 for all rows")

            # Ensure no NaNs in engineered features
            df_books[["recency_score", "rating_score", "page_score"]] = df_books[
                ["recency_score", "rating_score", "page_score"]
            ].fillna(0)

            # Create combined text feature for vectorization downstream.
            # Ensure all parts exist and are strings to avoid concatenation errors.
            for text_col in ["title", "description", "categories", "authors"]:
                if text_col not in df_books.columns:
                    df_books[text_col] = ""
            df_books["combined_text"] = (
                df_books["title"].astype(str)
                + " "
                + df_books["description"].astype(str)
                + " "
                + df_books["categories"].astype(str)
                + " "
                + df_books["authors"].astype(str)
            )

            # Clean combined_text using the utility
            df_books["combined_text"] = df_books["combined_text"].apply(self.clean_text)

            # Persist modified books
            books_dir = os.path.dirname(self.build_features_config.modified_books_data_filepath)
            os.makedirs(books_dir, exist_ok=True)
            df_books.to_csv(self.build_features_config.modified_books_data_filepath, index=False)
            logger.info("Saved modified books features to: %s", self.build_features_config.modified_books_data_filepath)
        except FileNotFoundError as e:
            logger.exception("Cleaned books file not found: %s", e)
            raise
        except Exception as e:
            logger.exception("Error while building book features: %s", e)
            raise

    def build_paper_features(self) -> None:
        """
        Build features for papers and save modified CSV.

        Steps:
        - Load cleaned papers CSV.
        - Convert Year and Citations to numeric and compute scaled recency and citations scores.
        - Create combined_text (SearchQuery + Title + Abstract + Authors) and clean it.
        - Persist modified CSV to configured path.
        """
        try:
            logger.info("Building paper features from: %s", self.data_cleaning_artifact.cleaned_papers_data_filepath)
            df_paper = pd.read_csv(self.data_cleaning_artifact.cleaned_papers_data_filepath)

            # Ensure expected columns exist; create empty columns and warn if missing.
            expected_cols = ["Year", "Citations", "SearchQuery", "Title", "Abstract", "Authors"]
            for c in expected_cols:
                if c not in df_paper.columns:
                    logger.warning("Expected column '%s' missing from cleaned papers CSV. Creating empty column.", c)
                    df_paper[c] = ""

            df_paper["Year"] = pd.to_numeric(df_paper["Year"], errors="coerce")
            scaler = MinMaxScaler()

            # Recency score for papers
            if df_paper["Year"].notna().any():
                valid_years = df_paper.loc[df_paper["Year"].notna(), "Year"].astype(float).values.reshape(-1, 1)
                df_paper.loc[df_paper["Year"].notna(), "recency_score"] = scaler.fit_transform(valid_years)
                logger.debug("Computed recency_score for %d papers", valid_years.shape[0])
            else:
                df_paper["recency_score"] = 0.0
                logger.warning("No valid Year values found in papers; recency_score set to 0 for all rows")

            # Citations score
            try:
                citations = df_paper["Citations"].fillna(0).astype(float).values.reshape(-1, 1)
                df_paper["citations_score"] = MinMaxScaler().fit_transform(citations)
            except Exception as e:
                logger.exception("Failed to compute citations_score: %s", e)
                df_paper["citations_score"] = 0.0
                logger.warning("Failed to compute citations_score; set to 0 for all rows")

            df_paper[["recency_score", "citations_score"]] = df_paper[["recency_score", "citations_score"]].fillna(0)

            # Build combined_text for papers
            for text_col in ["SearchQuery", "Title", "Abstract", "Authors"]:
                if text_col not in df_paper.columns:
                    df_paper[text_col] = ""
            df_paper["combined_text"] = (
                df_paper["SearchQuery"].astype(str)
                + " "
                + df_paper["Title"].astype(str)
                + " "
                + df_paper["Abstract"].astype(str)
                + " "
                + df_paper["Authors"].astype(str)
            )

            df_paper["combined_text"] = df_paper["combined_text"].apply(self.clean_text)

            # Persist modified papers
            papers_dir = os.path.dirname(self.build_features_config.modified_papers_data_filepath)
            os.makedirs(papers_dir, exist_ok=True)
            df_paper.to_csv(self.build_features_config.modified_papers_data_filepath, index=False)
            logger.info("Saved modified paper features to: %s", self.build_features_config.modified_papers_data_filepath)
        except FileNotFoundError as e:
            logger.exception("Cleaned papers file not found: %s", e)
            raise
        except Exception as e:
            logger.exception("Error while building paper features: %s", e)
            raise

    def initiate_build_features(self) -> BuildFeaturesArifact:
        """
        Execute feature building pipeline for books and papers and return artifact.

        Returns:
            BuildFeaturesArifact: artifact containing paths to modified CSVs.
        """
        try:
            logger.info("Starting feature build pipeline...")
            self.build_book_features()
            self.build_paper_features()
            logger.info("Feature build pipeline completed successfully")
            return BuildFeaturesArifact(
                modified_books_data_filepath=self.build_features_config.modified_books_data_filepath,
                modified_papers_data_filepath=self.build_features_config.modified_papers_data_filepath,
            )
        except Exception as e:
            logger.exception("Failed to complete feature build pipeline: %s", e)
            raise

def main():
    """
    Simple CLI/test entrypoint for this module.

    Note:
    - Uses BuildFeatureConfig() defaults. Ensure BuildFeatureConfig is configured
      (or replace with a manual instantiation) before running this script.
    """
    try:
        logger.info("Running build_features module as script for quick test")

        cleaning_artifact = DataCleaningArtifact(
            cleaned_books_data_filepath="data/raw/Ml_books.csv",
            cleaned_papers_data_filepath="data/raw/all_papers.csv",
        )
        build_config = BuildFeatureConfig()

        builder = BuildFeatures(build_config, cleaning_artifact)
        artifact = builder.initiate_build_features()
        logger.info("Build features artifact: %s", getattr(artifact, "__dict__", str(artifact)))
        print("Feature build completed.")
        print("Modified books file:", artifact.modified_books_data_filepath)
        print("Modified papers file:", artifact.modified_papers_data_filepath)
    except Exception as e:
        logger.exception("Error while testing build_features module: %s", e)
        print("Error during feature building:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()






