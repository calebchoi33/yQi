"""Centralized directory management utility for yQi evaluation platform."""

import os
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DirectoryManager:
    """Centralized directory management with date-based organization."""
    
    def __init__(self, base_dir: str = None):
        """Initialize with optional base directory."""
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(__file__))
    
    def ensure_directory(self, directory_path: str) -> str:
        """Ensure directory exists, create if it doesn't.
        
        Args:
            directory_path: Path to directory (absolute or relative to base_dir)
            
        Returns:
            Absolute path to the directory
        """
        if not os.path.isabs(directory_path):
            directory_path = os.path.join(self.base_dir, directory_path)
        
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
            logger.info(f"Created directory: {directory_path}")
        
        return directory_path
    
    def get_date_directory(self, base_dir: str, date: Optional[datetime] = None) -> str:
        """Get date-organized directory path.
        
        Args:
            base_dir: Base directory name (e.g., 'responses', 'benchmarks')
            date: Date to use (defaults to today)
            
        Returns:
            Path to date-organized directory
        """
        if date is None:
            date = datetime.now()
        
        date_folder = date.strftime("%Y-%m-%d")
        date_dir = os.path.join(self.base_dir, base_dir, date_folder)
        
        return self.ensure_directory(date_dir)
    
    def get_responses_directory(self, date: Optional[datetime] = None) -> str:
        """Get responses directory for given date."""
        return self.get_date_directory("responses", date)
    
    def get_benchmarks_directory(self, date: Optional[datetime] = None) -> str:
        """Get benchmarks directory for given date."""
        return self.get_date_directory("benchmarks", date)
    
    def get_batches_directory(self, date: Optional[datetime] = None) -> str:
        """Get batches directory for given date."""
        return self.get_date_directory("batches", date)
    
    def get_batch_results_directory(self, date: Optional[datetime] = None) -> str:
        """Get batch results directory for given date."""
        return self.get_date_directory("batch_results", date)
    
    def get_ratings_directory(self, run_id: Optional[str] = None) -> str:
        """Get ratings directory, optionally organized by run ID."""
        ratings_dir = self.ensure_directory("ratings")
        
        if run_id:
            run_dir = os.path.join(ratings_dir, run_id)
            return self.ensure_directory(run_dir)
        
        return ratings_dir
    
    def cleanup_empty_directories(self, directory: str) -> int:
        """Remove empty directories recursively.
        
        Args:
            directory: Directory to clean up
            
        Returns:
            Number of directories removed
        """
        removed_count = 0
        
        try:
            for root, dirs, files in os.walk(directory, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        if not os.listdir(dir_path):  # Directory is empty
                            os.rmdir(dir_path)
                            removed_count += 1
                            logger.info(f"Removed empty directory: {dir_path}")
                    except OSError:
                        # Directory not empty or permission error
                        pass
        except Exception as e:
            logger.error(f"Error during directory cleanup: {e}")
        
        return removed_count


# Global instance for convenience
directory_manager = DirectoryManager()
