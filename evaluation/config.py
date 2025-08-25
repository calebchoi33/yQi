"""Legacy configuration - DEPRECATED. Use core.config instead."""

# This file is deprecated. All configuration has been moved to core/config.py
# Import from there instead:
# from core.config import AppConfig

import warnings
warnings.warn(
    "config.py is deprecated. Use 'from core.config import AppConfig' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Legacy constants for backward compatibility
RESPONSES_DIR = "responses"
RESPONSES_FILE = "chatgpt_responses.json"
BENCHMARKS_DIR = "benchmarks"
MODEL_NAME = "gpt-4o"
MAX_TOKENS = 500
TEMPERATURE = 0.7
BATCH_SIZE = 5
MAX_RETRIES = 3
RETRY_DELAY = 5
