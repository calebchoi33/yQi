"""File management operations for responses and benchmarks.

This module handles all file I/O operations for the CM AGI Evaluation Platform,
including saving and loading response data and benchmark metrics to/from JSON files.
It manages directory creation, timestamped filenames, and finding the latest files.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, Optional
from core.config import AppConfig

# Initialize config for backward compatibility
_config = AppConfig()
RESPONSES_DIR = _config.storage.responses_dir
BENCHMARKS_DIR = _config.storage.benchmarks_dir
RESPONSES_FILE = _config.storage.responses_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("file_manager.log"),
        logging.StreamHandler()
    ]
)


def save_responses_to_json(responses_data: Dict) -> str:
    """Save responses to a timestamped JSON file in the responses directory.
    
    Creates the responses directory if it doesn't exist and generates a timestamped
    filename to prevent overwriting previous response files.
    
    Args:
        responses_data: Dictionary containing the response data to save
        
    Returns:
        str: Path to the saved JSON file
        
    Raises:
        IOError: If there's an issue creating the directory or writing the file
        TypeError: If the responses_data cannot be serialized to JSON
    """
    try:
        # Create responses directory if it doesn't exist
        if not os.path.exists(RESPONSES_DIR):
            logging.info(f"Creating responses directory: {RESPONSES_DIR}")
            os.makedirs(RESPONSES_DIR)
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"responses_{timestamp}.json"
        filepath = os.path.join(RESPONSES_DIR, filename)
        
        logging.info(f"Saving responses to {filepath}")
        with open(filepath, 'w') as f:
            json.dump(responses_data, f, indent=2)
        
        logging.info(f"Successfully saved responses to {filepath}")
        return filepath
    except IOError as e:
        logging.error(f"IO error saving responses: {str(e)}")
        raise
    except TypeError as e:
        logging.error(f"Type error when serializing responses to JSON: {str(e)}")
        raise


def save_benchmark_data(benchmark_data: Dict) -> str:
    """Save benchmark data to a timestamped JSON file in the benchmarks directory.
    
    Creates the benchmarks directory if it doesn't exist and uses the run_id from
    the benchmark data to create a unique filename.
    
    Args:
        benchmark_data: Dictionary containing the benchmark metrics to save
        
    Returns:
        str: Path to the saved benchmark JSON file
        
    Raises:
        IOError: If there's an issue creating the directory or writing the file
        KeyError: If the benchmark_data doesn't contain a run_id key
        TypeError: If the benchmark_data cannot be serialized to JSON
    """
    try:
        # Validate benchmark_data contains run_id
        if 'run_id' not in benchmark_data:
            logging.error("Benchmark data missing required 'run_id' key")
            raise KeyError("Benchmark data must contain a 'run_id' key")
            
        # Create benchmarks directory if it doesn't exist
        if not os.path.exists(BENCHMARKS_DIR):
            logging.info(f"Creating benchmarks directory: {BENCHMARKS_DIR}")
            os.makedirs(BENCHMARKS_DIR)
        
        # Create filename with timestamp
        filename = f"benchmark_{benchmark_data['run_id']}.json"
        filepath = os.path.join(BENCHMARKS_DIR, filename)
        
        logging.info(f"Saving benchmark data to {filepath}")
        with open(filepath, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        logging.info(f"Successfully saved benchmark data to {filepath}")
        return filepath
    except IOError as e:
        logging.error(f"IO error saving benchmark data: {str(e)}")
        raise
    except TypeError as e:
        logging.error(f"Type error when serializing benchmark data to JSON: {str(e)}")
        raise


def load_responses_from_json() -> Optional[Dict]:
    """Load the most recent responses from JSON file in the responses directory.
    
    Finds all response files in the responses directory, sorts them by filename
    (which includes a timestamp), and loads the most recent one.
    
    Returns:
        Dict: The loaded response data, or None if no files are found
        
    Raises:
        IOError: If there's an issue reading the file
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        if not os.path.exists(RESPONSES_DIR):
            logging.info(f"Responses directory {RESPONSES_DIR} does not exist")
            return None
        
        # Get all response files and find the most recent one
        response_files = [f for f in os.listdir(RESPONSES_DIR) if f.startswith('responses_') and f.endswith('.json')]
        if not response_files:
            logging.info("No response files found in directory")
            return None
        
        # Sort by filename (which includes timestamp) to get most recent
        response_files.sort(reverse=True)
        latest_file = os.path.join(RESPONSES_DIR, response_files[0])
        
        logging.info(f"Loading responses from {latest_file}")
        with open(latest_file, 'r') as f:
            data = json.load(f)
            logging.info(f"Successfully loaded responses from {latest_file}")
            return data
    except IOError as e:
        logging.error(f"IO error loading responses: {str(e)}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in response file: {str(e)}")
        raise


def get_latest_responses_file() -> Optional[str]:
    """Get the path to the most recent responses file for download.
    
    Returns:
        str: The full path to the most recent responses file, or None if no files are found
        
    Raises:
        IOError: If there's an issue accessing the directory
    """
    try:
        if not os.path.exists(RESPONSES_DIR):
            logging.info(f"Responses directory {RESPONSES_DIR} does not exist")
            return None
        
        response_files = [f for f in os.listdir(RESPONSES_DIR) if f.startswith('responses_') and f.endswith('.json')]
        if not response_files:
            logging.info("No response files found in directory")
            return None
        
        response_files.sort(reverse=True)
        latest_file = os.path.join(RESPONSES_DIR, response_files[0])
        logging.info(f"Latest responses file: {latest_file}")
        return latest_file
    except IOError as e:
        logging.error(f"IO error accessing responses directory: {str(e)}")
        raise


def get_latest_responses_filename() -> Optional[str]:
    """Get the filename of the most recent responses file.
    
    Returns:
        str: The filename (without path) of the most recent responses file, or None if no files are found
        
    Raises:
        IOError: If there's an issue accessing the directory
    """
    try:
        if not os.path.exists(RESPONSES_DIR):
            logging.info(f"Responses directory {RESPONSES_DIR} does not exist")
            return None
        
        response_files = [f for f in os.listdir(RESPONSES_DIR) if f.startswith('responses_') and f.endswith('.json')]
        if not response_files:
            logging.info("No response files found in directory")
            return None
        
        response_files.sort(reverse=True)
        latest_filename = response_files[0]
        logging.info(f"Latest responses filename: {latest_filename}")
        return latest_filename
    except IOError as e:
        logging.error(f"IO error accessing responses directory: {str(e)}")
        raise
