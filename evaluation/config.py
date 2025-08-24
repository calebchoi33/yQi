"""Configuration constants and default prompts for CM AGI Evaluation Platform.

This module contains all configuration settings and default prompts used throughout
the application. Centralizing these values makes it easier to modify application
behavior without changing code in multiple files.
"""

# Configuration constants

# Directory and file constants
RESPONSES_DIR = "responses"
RESPONSES_FILE = "chatgpt_responses.json"
BENCHMARKS_DIR = "benchmarks"

# API Configuration
MODEL_NAME = "gpt-4o"
MAX_TOKENS = 500
TEMPERATURE = 0.7
BATCH_SIZE = 5  # Number of prompts to send in a single API call

# Retry Configuration
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Default medical case prompts for TCM evaluation
