"""
Data loader for model1.

Provides helpers to load interim CSVs (books and papers) from the project's
data/interim folder. Functions return pandas.DataFrame by default or CSV
string when as_csv=True.

Usage:
    from src.models.model1.dataloader import get_books, get_papers
    df_books = get_books()
    csv_str = get_papers(as_csv=True)
"""
from typing import Tuple, Union
import os
import pandas as pd

from logger import get_logger

logger = get_logger(log_filename="dataloader.log")


def _project_root() -> str:
    """
    Return absolute path to the repository root based on this file location.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _interim_path(filename: str) -> str:
    """
    Build full path to a file in data/interim.

    Args:
        filename: name of the file inside data/interim

    Returns:
        Absolute file path string.
    """
    return os.path.join(_project_root(), "data", "interim", filename)


def load_csv_from_interim(filename: str, as_csv: bool = False) -> Union[pd.DataFrame, str]:
    """
    Load a CSV from data/interim.

    Args:
        filename: CSV filename inside data/interim (e.g. "modified_books.csv").
        as_csv: If True, return CSV content as a string. If False, return DataFrame.

    Returns:
        pandas.DataFrame or CSV string.

    Raises:
        FileNotFoundError: If the target file does not exist.
        pd.errors.EmptyDataError / other pandas exceptions may propagate.
    """
    path = _interim_path(filename)
    logger.debug("Attempting to load interim file: %s", path)

    if not os.path.exists(path):
        logger.error("Interim file not found: %s", path)
        raise FileNotFoundError(f"Interim file not found: {path}")

    try:
        df = pd.read_csv(path)
        logger.info("Loaded %s rows from %s", len(df), path)
        return df.to_csv(index=False) if as_csv else df
    except Exception as e:
        logger.exception("Failed to read CSV %s : %s", path, e)
        raise


def get_books(as_csv: bool = False) -> Union[pd.DataFrame, str]:
    """
    Load modified books CSV from data/interim.

    Args:
        as_csv: If True, return CSV string; otherwise DataFrame.

    Returns:
        DataFrame or CSV string with books data.
    """
    logger.debug("get_books called (as_csv=%s)", as_csv)
    return load_csv_from_interim("modified_books.csv", as_csv=as_csv)


def get_papers(as_csv: bool = False) -> Union[pd.DataFrame, str]:
    """
    Load modified papers CSV from data/interim.

    Args:
        as_csv: If True, return CSV string; otherwise DataFrame.

    Returns:
        DataFrame or CSV string with papers data.
    """
    logger.debug("get_papers called (as_csv=%s)", as_csv)
    return load_csv_from_interim("modified_papers.csv", as_csv=as_csv)


def get_all_interim(as_csv: bool = False) -> Tuple[Union[pd.DataFrame, str], Union[pd.DataFrame, str]]:
    """
    Load both books and papers interim files.

    Args:
        as_csv: If True, both returned values are CSV strings; otherwise DataFrames.

    Returns:
        Tuple (books, papers)
    """
    logger.debug("get_all_interim called (as_csv=%s)", as_csv)
    books = get_books(as_csv=as_csv)
    papers = get_papers(as_csv=as_csv)
    return books, papers


def main():
    """
    Quick local test for the dataloader module.
    Prints shapes (or CSV preview) for both interim files.
    """
    try:
        logger.info("Running dataloader main test")
        books_df = get_books()
        papers_df = get_papers()

        print("Books:", type(books_df), getattr(books_df, "shape", None))
        print("Papers:", type(papers_df), getattr(papers_df, "shape", None))

        # also show a short CSV preview for manual inspection
        csv_preview = get_books(as_csv=True)
        print("\nBooks CSV preview (first 300 chars):\n", csv_preview[:300])
    except FileNotFoundError as e:
        logger.error("Data files missing for dataloader test: %s", e)
        print("Missing data file:", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error in dataloader main: %s", e)
        print("Error while loading interim data:", e)
        raise


if __name__ == "__main__":
    main()