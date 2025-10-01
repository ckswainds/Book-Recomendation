"""
Custom exceptions for the Book Recommendation System.
"""

class BookRecommenderError(Exception):
    """
    Base class for all custom exceptions in the book recommendation system.
    All other custom exceptions should inherit from this class.
    """
    def __init__(self, message="An error occurred in the book recommendation system."):
        self.message = message
        super().__init__(self.message)
        
    def __str__(self):
        return f"[{self.__class__.__name__}]: {self.message}"


# --- Data Loading and Preprocessing Exceptions ---

class DataLoadError(BookRecommenderError):
    """Raised when data loading fails (e.g., file corruption, wrong format)."""
    pass

class MissingDataFieldError(DataLoadError):
    """Raised when essential columns (e.g., 'book_id', 'title') are missing."""
    def __init__(self, required_field, message=None):
        if message is None:
            message = f"Required field '{required_field}' is missing in the dataset."
        super().__init__(message)
        self.required_field = required_field


# --- Model Training and Inference Exceptions ---

class ModelTrainingError(BookRecommenderError):
    """Raised when the model fails to train correctly."""
    pass

class InsufficientTrainingDataError(ModelTrainingError):
    """Raised when the dataset is too small or sparse to train the model."""
    def __init__(self, count, min_required=100, message=None):
        if message is None:
            message = (f"Insufficient data for training. Found {count} usable records, "
                       f"but need at least {min_required}.")
        super().__init__(message)
        self.count = count
        self.min_required = min_required

class ModelNotTrainedError(BookRecommenderError):
    """Raised when a prediction is attempted before the model is trained/loaded."""
    def __init__(self, message="Model has not been trained or loaded. Cannot make recommendations."):
        super().__init__(message)


# --- Recommendation/User Interaction Exceptions ---

class InvalidUserError(BookRecommenderError):
    """Raised when a user ID does not exist in the system."""
    def __init__(self, user_id, message=None):
        if message is None:
            message = f"User ID '{user_id}' not found in the user database."
        super().__init__(message)
        self.user_id = user_id

class NoRecommendationsFoundError(BookRecommenderError):
    """Raised when the model successfully runs but returns zero recommendations 
       (e.g., for a cold-start user)."""
    def __init__(self, user_id, message=None):
        if message is None:
            message = f"Could not generate any recommendations for user ID '{user_id}'."
        super().__init__(message)
        self.user_id = user_id