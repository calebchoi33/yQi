"""Centralized configuration management for yQi evaluation platform."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class APIConfig:
    """API configuration settings."""
    model_name: str = "gpt-4o"
    max_tokens: int = 500
    temperature: float = 0.7
    batch_size: int = 5
    max_retries: int = 3
    retry_delay: int = 5
    api_key: Optional[str] = os.getenv('OPENAI_API_KEY')


@dataclass
class StorageConfig:
    """File storage configuration."""
    responses_dir: str = "responses"
    benchmarks_dir: str = "benchmarks"
    ratings_dir: str = "ratings"
    responses_file: str = "chatgpt_responses.json"


class AppConfig:
    """Main application configuration."""
    
    def __init__(self):
        self.api = APIConfig()
        self.storage = StorageConfig()
    
    @property
    def is_api_configured(self) -> bool:
        """Check if API is properly configured."""
        return bool(self.api.api_key)
