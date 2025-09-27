"""
Batch Job Management System for yQi Evaluation Platform

This module handles OpenAI Batch API job lifecycle management including:
- Job persistence and tracking
- Status monitoring and updates
- Result retrieval and processing
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from openai import OpenAI

from api_client import (
    check_batch_status, retrieve_batch_results, 
    process_batch_results, process_prompts_batch
)
from file_manager import save_responses_to_json, save_benchmark_data


class BatchJobManager:
    """Manages OpenAI Batch API jobs for the yQi platform."""
    
    def __init__(self):
        self.jobs_file = os.path.join(os.path.dirname(__file__), "batch_jobs.json")
        self.jobs = self._load_jobs()
    
    def _load_jobs(self) -> Dict[str, Dict]:
        """Load batch jobs from persistent storage."""
        if os.path.exists(self.jobs_file):
            try:
                with open(self.jobs_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading batch jobs: {e}")
                return {}
        return {}
    
    def _save_jobs(self):
        """Save batch jobs to persistent storage."""
        try:
            # Custom JSON encoder to handle non-serializable objects
            def json_serializer(obj):
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                elif hasattr(obj, '_asdict'):
                    return obj._asdict()
                else:
                    return str(obj)
            
            with open(self.jobs_file, 'w', encoding='utf-8') as f:
                json.dump(self.jobs, f, indent=2, ensure_ascii=False, default=json_serializer)
        except Exception as e:
            logging.error(f"Error saving batch jobs: {e}")
    
    def add_job(self, batch_job_id: str, batch_info: Dict):
        """Add a new batch job to tracking."""
        self.jobs[batch_job_id] = batch_info
        self._save_jobs()
        logging.info(f"Added batch job {batch_job_id} to tracking")
    
    def get_active_jobs(self) -> List[Dict]:
        """Get list of active (non-completed) batch jobs."""
        active_jobs = []
        for job_id, job_info in self.jobs.items():
            if job_info.get('status') not in ['completed', 'failed', 'cancelled']:
                job_info['batch_job_id'] = job_id
                active_jobs.append(job_info)
        return active_jobs
    
    def get_all_jobs(self) -> List[Dict]:
        """Get list of all batch jobs."""
        all_jobs = []
        for job_id, job_info in self.jobs.items():
            job_info['batch_job_id'] = job_id
            all_jobs.append(job_info)
        return sorted(all_jobs, key=lambda x: x.get('submitted_at', ''), reverse=True)
    
    def update_job_status(self, batch_job_id: str, api_key: str) -> Dict:
        """Update job status by checking with OpenAI API."""
        if batch_job_id not in self.jobs:
            raise ValueError(f"Job {batch_job_id} not found")
        
        try:
            client = OpenAI(api_key=api_key)
            status_info = check_batch_status(client, batch_job_id)
            
            # Convert request_counts to dict if it's a BatchRequestCounts object
            request_counts = status_info.get('request_counts')
            if request_counts and hasattr(request_counts, '__dict__'):
                request_counts = request_counts.__dict__
            
            # Update job info
            self.jobs[batch_job_id].update({
                'status': status_info['status'],
                'created_at': status_info.get('created_at'),
                'completed_at': status_info.get('completed_at'),
                'failed_at': status_info.get('failed_at'),
                'request_counts': request_counts,
                'output_file_id': status_info.get('output_file_id'),
                'error_file_id': status_info.get('error_file_id'),
                'last_checked': datetime.now().isoformat()
            })
            
            self._save_jobs()
            logging.info(f"Updated status for job {batch_job_id}: {status_info['status']}")
            return status_info
            
        except Exception as e:
            logging.error(f"Error updating job status: {e}")
            raise
    
    def download_results(self, batch_job_id: str, api_key: str) -> Tuple[Dict, Dict]:
        """Download and process batch job results."""
        if batch_job_id not in self.jobs:
            raise ValueError(f"Job {batch_job_id} not found")
        
        job_info = self.jobs[batch_job_id]
        
        # Update status first
        self.update_job_status(batch_job_id, api_key)
        
        if job_info['status'] != 'completed':
            raise ValueError(f"Job not completed. Status: {job_info['status']}")
        
        try:
            client = OpenAI(api_key=api_key)
            
            # Download results
            results_file_path = retrieve_batch_results(
                client, batch_job_id, job_info['run_id']
            )
            
            # Process results into yQi format
            responses_data, benchmark_data = process_batch_results(
                results_file_path=results_file_path,
                original_prompts=job_info['prompts'],
                system_prompt=job_info['system_prompt'],
                run_id=job_info['run_id']
            )
            
            # Save to standard yQi format
            responses_filepath = save_responses_to_json(responses_data)
            benchmark_filepath = save_benchmark_data(benchmark_data)
            
            # Update job info
            self.jobs[batch_job_id].update({
                'status': 'downloaded',
                'responses_file': responses_filepath,
                'benchmark_file': benchmark_filepath,
                'downloaded_at': datetime.now().isoformat()
            })
            
            self._save_jobs()
            
            logging.info(f"Downloaded and processed results for job {batch_job_id}")
            return responses_data, benchmark_data
            
        except Exception as e:
            logging.error(f"Error downloading results: {e}")
            raise
    
    def submit_batch_job(self, api_key: str, system_prompt: str, prompts: List[str], 
                        model: str = "gpt-4o-mini", temperature: float = 0.7, 
                        max_tokens: int = 500) -> str:
        """Submit a new batch job and add to tracking."""
        try:
            batch_job_id, batch_info = process_prompts_batch(
                api_key=api_key,
                system_prompt=system_prompt,
                prompts=prompts,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Add to tracking
            self.add_job(batch_job_id, batch_info)
            
            return batch_job_id
            
        except Exception as e:
            logging.error(f"Error submitting batch job: {e}")
            raise
    
    def remove_job(self, batch_job_id: str):
        """Remove a job from tracking."""
        if batch_job_id in self.jobs:
            del self.jobs[batch_job_id]
            self._save_jobs()
            logging.info(f"Removed job {batch_job_id} from tracking")
    
    def cleanup_old_jobs(self, days_old: int = 30):
        """Remove jobs older than specified days."""
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        jobs_to_remove = []
        
        for job_id, job_info in self.jobs.items():
            submitted_at = job_info.get('submitted_at')
            if submitted_at:
                try:
                    job_date = datetime.fromisoformat(submitted_at)
                    if job_date < cutoff_date:
                        jobs_to_remove.append(job_id)
                except ValueError:
                    continue
        
        for job_id in jobs_to_remove:
            self.remove_job(job_id)
        
        if jobs_to_remove:
            logging.info(f"Cleaned up {len(jobs_to_remove)} old batch jobs")
        
        return len(jobs_to_remove)


# Global batch manager instance
batch_manager = BatchJobManager()
