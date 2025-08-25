"""File-based rating system for yQi evaluation platform."""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from core.config import AppConfig


class RatingManager:
    """Manages ratings using local JSON file storage."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.ratings_dir = config.storage.ratings_dir
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure ratings directory exists."""
        os.makedirs(self.ratings_dir, exist_ok=True)
    
    def _ensure_run_directory(self, run_id: str):
        """Ensure directory for specific run exists."""
        run_dir = os.path.join(self.ratings_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    
    def save_rating(self, run_id: str, prompt_number: int, rating_data: Dict[str, Any]) -> str:
        """Save a rating to a JSON file organized by run."""
        rating_id = f"{run_id}_{prompt_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        rating_record = {
            "id": rating_id,
            "run_id": run_id,
            "prompt_number": prompt_number,
            "timestamp": datetime.now().isoformat(),
            **rating_data
        }
        
        # Create run-specific directory
        run_dir = self._ensure_run_directory(run_id)
        
        filename = f"rating_{rating_id}.json"
        filepath = os.path.join(run_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(rating_record, f, indent=2, ensure_ascii=False)
        
        return rating_id
    
    def get_ratings_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all ratings for a specific run."""
        ratings = []
        
        # Check both old format (flat) and new format (organized by run)
        run_dir = os.path.join(self.ratings_dir, run_id)
        
        # New format: check run-specific directory
        if os.path.exists(run_dir):
            for filename in os.listdir(run_dir):
                if filename.startswith(f"rating_{run_id}_") and filename.endswith('.json'):
                    filepath = os.path.join(run_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            rating = json.load(f)
                            ratings.append(rating)
                    except Exception as e:
                        print(f"Error loading rating file {filename}: {e}")
        
        # Legacy format: check flat directory for backward compatibility
        if os.path.exists(self.ratings_dir):
            for filename in os.listdir(self.ratings_dir):
                if filename.startswith(f"rating_{run_id}_") and filename.endswith('.json'):
                    filepath = os.path.join(self.ratings_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            rating = json.load(f)
                            ratings.append(rating)
                    except Exception as e:
                        print(f"Error loading rating file {filename}: {e}")
        
        # Sort by prompt number
        ratings.sort(key=lambda x: x.get('prompt_number', 0))
        return ratings
    
    def get_all_ratings(self) -> List[Dict[str, Any]]:
        """Get all ratings across all runs."""
        ratings = []
        
        if not os.path.exists(self.ratings_dir):
            return ratings
        
        # Check both organized directories and flat files for backward compatibility
        for item in os.listdir(self.ratings_dir):
            item_path = os.path.join(self.ratings_dir, item)
            
            if os.path.isdir(item_path):
                # New format: run-specific directories
                for filename in os.listdir(item_path):
                    if filename.startswith('rating_') and filename.endswith('.json'):
                        filepath = os.path.join(item_path, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                rating = json.load(f)
                                ratings.append(rating)
                        except Exception as e:
                            print(f"Error loading rating file {filename}: {e}")
            elif item.startswith('rating_') and item.endswith('.json'):
                # Legacy format: flat files
                try:
                    with open(item_path, 'r', encoding='utf-8') as f:
                        rating = json.load(f)
                        ratings.append(rating)
                except Exception as e:
                    print(f"Error loading rating file {item}: {e}")
        
        return ratings
    
    def get_available_runs(self) -> List[Dict[str, Any]]:
        """Get list of runs that have responses available for rating."""
        runs = []
        
        # Check responses directory for available runs
        responses_dir = self.config.storage.responses_dir
        if not os.path.exists(responses_dir):
            return runs
        
        # Search both root directory and date-organized subdirectories
        def scan_directory(directory):
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                
                if os.path.isdir(item_path):
                    # Recursively scan subdirectories (date folders)
                    scan_directory(item_path)
                elif item.startswith('responses_') and item.endswith('.json'):
                    try:
                        with open(item_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        # Extract run ID from filename
                        run_id = item.replace('responses_', '').replace('.json', '')
                        
                        # Get existing ratings for this run
                        existing_ratings = self.get_ratings_for_run(run_id)
                        rated_prompts = {r['prompt_number'] for r in existing_ratings}
                        
                        run_info = {
                            'run_id': run_id,
                            'timestamp': data.get('timestamp'),
                            'total_prompts': data.get('total_prompts', 0),
                            'rated_count': len(existing_ratings),
                            'unrated_count': data.get('total_prompts', 0) - len(rated_prompts),
                            'responses': data.get('responses', []),
                            'processing_mode': data.get('processing_mode', 'realtime'),
                            'batch_processing': data.get('batch_processing', False)
                        }
                        runs.append(run_info)
                        
                    except Exception as e:
                        print(f"Error loading response file {item}: {e}")
        
        scan_directory(responses_dir)
        
        # Sort by timestamp (newest first)
        runs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return runs
    
    def get_rating_statistics(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Calculate rating statistics."""
        if run_id:
            ratings = self.get_ratings_for_run(run_id)
        else:
            ratings = self.get_all_ratings()
        
        if not ratings:
            return {}
        
        # Calculate basic statistics
        overall_ratings = [r.get('overall_rating') for r in ratings if r.get('overall_rating')]
        
        if not overall_ratings:
            return {}
        
        stats = {
            'total_ratings': len(ratings),
            'average_overall_rating': sum(overall_ratings) / len(overall_ratings),
            'rating_distribution': {},
            'criteria_averages': {}
        }
        
        # Rating distribution
        for rating in overall_ratings:
            stats['rating_distribution'][rating] = stats['rating_distribution'].get(rating, 0) + 1
        
        # Criteria averages
        criteria_fields = ['accuracy_rating', 'completeness_rating', 'clarity_rating', 
                         'clinical_relevance_rating', 'safety_rating']
        
        for field in criteria_fields:
            values = [r.get(field) for r in ratings if r.get(field) is not None]
            if values:
                stats['criteria_averages'][field] = sum(values) / len(values)
        
        return stats
    
    def delete_run(self, run_id: str) -> bool:
        """Delete all data for a specific run including responses, benchmarks, and ratings."""
        try:
            deleted_items = []
            
            # Delete ratings (both old flat format and new organized format)
            run_ratings_dir = os.path.join(self.ratings_dir, run_id)
            if os.path.exists(run_ratings_dir):
                import shutil
                shutil.rmtree(run_ratings_dir)
                deleted_items.append(f"ratings directory: {run_ratings_dir}")
            
            # Delete legacy flat rating files
            if os.path.exists(self.ratings_dir):
                for filename in os.listdir(self.ratings_dir):
                    if filename.startswith(f"rating_{run_id}_") and filename.endswith('.json'):
                        filepath = os.path.join(self.ratings_dir, filename)
                        os.remove(filepath)
                        deleted_items.append(f"rating file: {filename}")
            
            # Delete responses file
            responses_dir = self.config.storage.responses_dir
            responses_file = os.path.join(responses_dir, f"responses_{run_id}.json")
            if os.path.exists(responses_file):
                os.remove(responses_file)
                deleted_items.append(f"responses file: responses_{run_id}.json")
            
            # Delete benchmark file
            benchmarks_dir = self.config.storage.benchmarks_dir
            benchmark_file = os.path.join(benchmarks_dir, f"benchmark_{run_id}.json")
            if os.path.exists(benchmark_file):
                os.remove(benchmark_file)
                deleted_items.append(f"benchmark file: benchmark_{run_id}.json")
            
            if deleted_items:
                print(f"Successfully deleted run {run_id}:")
                for item in deleted_items:
                    print(f"  - {item}")
                return True
            else:
                print(f"No files found for run {run_id}")
                return False
                
        except Exception as e:
            print(f"Error deleting run {run_id}: {e}")
            return False
