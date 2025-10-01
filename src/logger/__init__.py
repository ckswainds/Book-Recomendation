import logging
import sys
import os
from pathlib import Path

# Define the relative path for the log directory
LOG_DIR = "logs"
DEFAULT_LOG_FILENAME = "recommender.log"

def get_logger(name: str = 'book_recommender', 
               level=logging.INFO, 
               log_filename: str = DEFAULT_LOG_FILENAME):
    """
    Sets up and returns a configured logger instance, ensuring the log file 
    is placed inside the 'logs' directory.

    Args:
        name (str): The name of the logger (usually the module name, or a project name).
        level (int): The minimum logging level to output.
        log_filename (str): The name of the log file (e.g., 'recommender.log').

    Returns:
        logging.Logger: The configured logger instance.
    """
    
    # 1. Setup the Log Directory and File Path
    
    # Create the full path for the log directory 
    # Path('.') refers to the current working directory (usually the project root)
    log_dir_path = Path('.') / LOG_DIR
    
    # Create the directory if it does not exist
    try:
        log_dir_path.mkdir(exist_ok=True)
    except Exception as e:
        # Fallback if directory creation fails (e.g., permission issues)
        print(f"Warning: Could not create log directory '{log_dir_path}'. Logs will only go to console. Error: {e}")
        log_file_path = None # Set path to None to skip file logging
    else:
        # Define the full path to the log file
        log_file_path = log_dir_path / log_filename
        # Convert to string for the logging module
        log_file_path = str(log_file_path)

    # 2. Get the logger instance
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False 

    # 3. Define the formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Check if the logger already has handlers
    if not logger.handlers:
        
        # Console Handler (always add)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler (add only if path creation was successful)
        if log_file_path:
            try:
                file_handler = logging.FileHandler(log_file_path, mode='a')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                print(f"Logging output also directed to: {log_file_path}")
            except Exception as e:
                # Log a warning if file handler setup fails
                logger.warning(f"Could not set up file logger at '{log_file_path}'. Error: {e}")
                
    return logger

# Example usage (for testing this file directly):
if __name__ == '__main__':
    
    app_logger = get_logger(
        name='MainAppRunner', 
        level=logging.DEBUG, 
        log_filename='main_app.log'
    )
    
    app_logger.info("Application setup is complete.")
    app_logger.error("Test error to ensure it appears in the log file.")
    app_logger.debug("Test debug message.")
    
   
    db_logger = get_logger(
        name='DatabaseModule',
        log_filename='database_events.log' 
    )
    db_logger.info("Database connection established.")