"""OpenAI API client for ChatGPT interactions with benchmarking.

This module handles all interactions with the OpenAI API, including sending prompts,
processing responses, and collecting detailed benchmarking metrics. It provides
real-time progress tracking through Streamlit's UI components.

Implements asynchronous processing for improved performance when handling multiple prompts.
"""

import time
import logging
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Tuple, Any
import openai
from openai import OpenAI, AsyncOpenAI
import streamlit as st
from core.config import AppConfig

# Initialize config for backward compatibility
_config = AppConfig()
MODEL_NAME = _config.api.model_name
MAX_TOKENS = _config.api.max_tokens
TEMPERATURE = _config.api.temperature
MAX_RETRIES = _config.api.max_retries
RETRY_DELAY = _config.api.retry_delay
BATCH_SIZE = _config.api.batch_size

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_client.log"),
        logging.StreamHandler()
    ]
)


async def process_prompt(client: AsyncOpenAI, prompt: str, prompt_number: int, batch_idx: int, system_prompt: str = None) -> Dict[str, Any]:
    """Process a single prompt asynchronously.
    
    Args:
        client: AsyncOpenAI client instance
        prompt: Text prompt to send to the API
        prompt_number: Index of the prompt (1-based)
        batch_idx: Index of the batch this prompt belongs to
        system_prompt: Optional system prompt to provide context
        
    Returns:
        Dictionary with response data and metrics
    """
    prompt_start_time = time.time()
    retry_count = 0
    success = False
    result = {}
    
    while not success and retry_count <= MAX_RETRIES:
        try:
            # Build messages array with optional system prompt
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            
            prompt_end_time = time.time()
            prompt_duration = prompt_end_time - prompt_start_time
            success = True
            
            logging.info(f"Successfully processed prompt {prompt_number}/{prompt_number}")
            
            result = {
                "success": True,
                "response": response,
                "duration": prompt_duration,
                "prompt_number": prompt_number,
                "prompt": prompt,
                "batch_idx": batch_idx,
                "error": None
            }
            
        except openai.RateLimitError as e:
            retry_count += 1
            if retry_count <= MAX_RETRIES:
                wait_time = RETRY_DELAY * retry_count
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds... (Attempt {retry_count}/{MAX_RETRIES})")
                await asyncio.sleep(wait_time)
            else:
                prompt_end_time = time.time()
                prompt_duration = prompt_end_time - prompt_start_time
                result = {
                    "success": False,
                    "response": None,
                    "duration": prompt_duration,
                    "prompt_number": prompt_number,
                    "prompt": prompt,
                    "batch_idx": batch_idx,
                    "error": "Rate limit exceeded. Maximum retries exceeded."
                }
                break
                
        except openai.APIConnectionError as e:
            retry_count += 1
            if retry_count <= MAX_RETRIES:
                wait_time = RETRY_DELAY * retry_count
                logging.warning(f"Connection error. Retrying in {wait_time} seconds... (Attempt {retry_count}/{MAX_RETRIES})")
                await asyncio.sleep(wait_time)
            else:
                prompt_end_time = time.time()
                prompt_duration = prompt_end_time - prompt_start_time
                result = {
                    "success": False,
                    "response": None,
                    "duration": prompt_duration,
                    "prompt_number": prompt_number,
                    "prompt": prompt,
                    "batch_idx": batch_idx,
                    "error": "Connection error. Maximum retries exceeded."
                }
                break
                
        except Exception as e:
            prompt_end_time = time.time()
            prompt_duration = prompt_end_time - prompt_start_time
            logging.error(f"Error processing prompt {prompt_number}: {str(e)}")
            result = {
                "success": False,
                "response": None,
                "duration": prompt_duration,
                "prompt_number": prompt_number,
                "prompt": prompt,
                "batch_idx": batch_idx,
                "error": str(e)
            }
            break
    
    return result


async def process_batch(client: AsyncOpenAI, batch: List[str], batch_idx: int, start_idx: int, system_prompt: str = None) -> List[Dict[str, Any]]:
    """Process a batch of prompts concurrently.
    
    Args:
        client: AsyncOpenAI client instance
        batch: List of prompts to process in this batch
        batch_idx: Index of the current batch
        start_idx: Starting index for prompt numbering
        system_prompt: Optional system prompt to provide context
        
    Returns:
        List of results for each prompt in the batch
    """
    tasks = []
    for i, prompt in enumerate(batch):
        prompt_number = start_idx + i + 1
        tasks.append(process_prompt(client, prompt, prompt_number, batch_idx, system_prompt))
    
    return await asyncio.gather(*tasks)


def send_prompts_to_chatgpt(prompts: List[str], api_key: str, system_prompt: str = None) -> Tuple[Dict, Dict]:
    """Send a series of prompts to ChatGPT API and collect responses with benchmarking.
    
    This function processes prompts in batches, with each batch being processed
    concurrently using asyncio. It collects both the response content and detailed
    performance metrics. It displays a progress bar and status updates in the Streamlit UI.
    
    Args:
        prompts: List of text prompts to send to the API
        api_key: OpenAI API key for authentication
        system_prompt: Optional system prompt to provide context for all prompts
        
    Returns:
        Tuple containing:
            - responses_data: Dictionary with all prompt-response pairs and metadata
            - benchmark_data: Dictionary with detailed performance metrics
    """
    # Define the async main function that will be run by the event loop
    async def async_main():
        # Initialize AsyncOpenAI client with explicit parameters
        async_client = AsyncOpenAI(
            api_key=api_key,
            timeout=60.0,
            max_retries=0  # We handle retries manually
        )
        
        # Initialize benchmarking timestamps and metrics
        start_time = time.time()
        benchmark_start = datetime.now()
        
        responses_data = {
            "timestamp": benchmark_start.isoformat(),
            "total_prompts": len(prompts),
            "responses": []
        }
        
        benchmark_data = {
            "run_id": benchmark_start.strftime("%Y%m%d_%H%M%S"),
            "start_time": benchmark_start.isoformat(),
            "total_prompts": len(prompts),
            "model": MODEL_NAME,
            "max_tokens_per_prompt": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "prompt_processing_times": [],
            "total_tokens_used": 0,
            "successful_prompts": 0,
            "failed_prompts": 0,
            "errors": []
        }
        
        # Chunk prompts by BATCH_SIZE
        def _chunk(items: List[str], size: int) -> List[List[str]]:
            return [items[i:i + size] for i in range(0, len(items), size)]
        
        batches = _chunk(prompts, max(1, BATCH_SIZE))
        total = len(prompts)
        processed = 0
        
        # Process each batch
        for batch_idx, batch in enumerate(batches, start=1):
            status_text.text(f"Processing batch {batch_idx}/{len(batches)} with {len(batch)} prompts...")
            logging.info(f"Starting batch {batch_idx}/{len(batches)} with {len(batch)} prompts")
            
            # Process the batch concurrently
            batch_results = await process_batch(async_client, batch, batch_idx, processed, system_prompt)
            
            # Process results from the batch
            for result in batch_results:
                if result["success"]:
                    response = result["response"]
                    
                    response_data = {
                        "prompt_number": result["prompt_number"],
                        "prompt": result["prompt"],
                        "response": response.choices[0].message.content,
                        "tokens_used": response.usage.total_tokens,
                        "timestamp": datetime.now().isoformat(),
                        "batch_index": result["batch_idx"],
                    }
                    
                    responses_data["responses"].append(response_data)
                    
                    benchmark_data["prompt_processing_times"].append({
                        "prompt_number": result["prompt_number"],
                        "duration_seconds": result["duration"],
                        "tokens_used": response.usage.total_tokens,
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(response.usage, "completion_tokens", None),
                        "batch_index": result["batch_idx"],
                    })
                    benchmark_data["total_tokens_used"] += response.usage.total_tokens
                    benchmark_data["successful_prompts"] += 1
                else:
                    error_message = result["error"]
                    logging.error(f"Failed to process prompt {result['prompt_number']}: {error_message}")
                    
                    response_data = {
                        "prompt_number": result["prompt_number"],
                        "prompt": result["prompt"],
                        "response": f"Error: {error_message}",
                        "tokens_used": 0,
                        "timestamp": datetime.now().isoformat(),
                        "batch_index": result["batch_idx"],
                    }
                    responses_data["responses"].append(response_data)
                    
                    benchmark_data["prompt_processing_times"].append({
                        "prompt_number": result["prompt_number"],
                        "duration_seconds": result["duration"],
                        "tokens_used": 0,
                        "error": error_message,
                        "batch_index": result["batch_idx"],
                    })
                    benchmark_data["failed_prompts"] += 1
                    benchmark_data["errors"].append({
                        "prompt_number": result["prompt_number"],
                        "error": error_message,
                        "timestamp": datetime.now().isoformat(),
                        "batch_index": result["batch_idx"],
                    })
                
                processed += 1
                progress_bar.progress(processed / total if total else 1)
        
        # Complete benchmarking
        end_time = time.time()
        total_duration = end_time - start_time
        
        benchmark_data.update({
            "end_time": datetime.now().isoformat(),
            "total_duration_seconds": total_duration,
            "average_time_per_prompt": total_duration / len(prompts) if prompts else 0,
            "tokens_per_second": benchmark_data["total_tokens_used"] / total_duration if total_duration > 0 else 0,
            "success_rate": benchmark_data["successful_prompts"] / len(prompts) if prompts else 0
        })
        
        return responses_data, benchmark_data
    # Create Streamlit UI elements
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run the async function using asyncio
    try:
        # Run the async main function and get the results
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        responses_data, benchmark_data = loop.run_until_complete(async_main())
        loop.close()
        
        status_text.text("All prompts processed!")
        return responses_data, benchmark_data
        
    except Exception as e:
        logging.error(f"Error in async processing: {str(e)}")
        status_text.text(f"Error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        raise


# Batch API Functions
def create_batch_requests(system_prompt: str, prompts: List[str], model: str = "gpt-4o-mini", 
                         temperature: float = 0.7, max_tokens: int = 500) -> List[Dict]:
    """Create batch request objects for OpenAI Batch API.
    
    Args:
        system_prompt: System prompt to use for all requests
        prompts: List of user prompts to process
        model: OpenAI model to use
        temperature: Temperature setting
        max_tokens: Maximum tokens per response
        
    Returns:
        List of batch request objects in the required format
    """
    batch_requests = []
    
    for i, prompt in enumerate(prompts):
        request = {
            "custom_id": f"request-{i}",
            "method": "POST", 
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            }
        }
        batch_requests.append(request)
    
    return batch_requests


def save_batch_file(batch_requests: List[Dict], run_id: str) -> str:
    """Save batch requests to JSONL file.
    
    Args:
        batch_requests: List of batch request objects
        run_id: Unique identifier for this batch
        
    Returns:
        Path to the created JSONL file
    """
    from core.directory_manager import directory_manager
    
    # Get date-organized batch directory
    batch_dir = directory_manager.get_batches_directory()
    
    # Create JSONL file
    filename = f"batch_{run_id}.jsonl"
    filepath = os.path.join(batch_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for request in batch_requests:
            f.write(json.dumps(request) + '\n')
    
    logging.info(f"Batch file saved to {filepath}")
    return filepath


def submit_batch_job(client, batch_file_path: str) -> str:
    """Submit a batch job to OpenAI.
    
    Args:
        client: OpenAI client instance
        batch_file_path: Path to the JSONL batch file
        
    Returns:
        Batch job ID
    """
    # Upload the batch file
    with open(batch_file_path, "rb") as f:
        batch_file = client.files.create(
            file=f,
            purpose="batch"
        )
    
    # Create the batch job
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    logging.info(f"Batch job created with ID: {batch_job.id}")
    return batch_job.id


def check_batch_status(client, batch_job_id: str) -> Dict:
    """Check the status of a batch job.
    
    Args:
        client: OpenAI client instance
        batch_job_id: ID of the batch job
        
    Returns:
        Dictionary with batch job status information
    """
    batch_job = client.batches.retrieve(batch_job_id)
    
    return {
        "id": batch_job.id,
        "status": batch_job.status,
        "created_at": batch_job.created_at,
        "completed_at": batch_job.completed_at,
        "failed_at": batch_job.failed_at,
        "request_counts": batch_job.request_counts,
        "output_file_id": batch_job.output_file_id,
        "error_file_id": batch_job.error_file_id
    }


def retrieve_batch_results(client, batch_job_id: str, run_id: str) -> str:
    """Retrieve and save batch job results.
    
    Args:
        client: OpenAI client instance
        batch_job_id: ID of the completed batch job
        run_id: Unique identifier for this batch
        
    Returns:
        Path to the saved results file
    """
    from core.directory_manager import directory_manager
    
    # Get batch job info
    batch_job = client.batches.retrieve(batch_job_id)
    
    if batch_job.status != "completed":
        raise ValueError(f"Batch job not completed. Status: {batch_job.status}")
    
    # Download results
    result_file_id = batch_job.output_file_id
    result_content = client.files.content(result_file_id).content
    
    # Get date-organized results directory
    results_dir = directory_manager.get_batch_results_directory()
    
    # Save results file
    filename = f"batch_results_{run_id}.jsonl"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'wb') as f:
        f.write(result_content)
    
    logging.info(f"Batch results saved to {filepath}")
    return filepath


def process_batch_results(results_file_path: str, original_prompts: List[str], 
                         system_prompt: str, run_id: str) -> Tuple[Dict, Dict]:
    """Process batch results into the standard yQi format.
    
    Args:
        results_file_path: Path to the batch results JSONL file
        original_prompts: Original list of prompts
        system_prompt: System prompt used
        run_id: Unique identifier for this batch
        
    Returns:
        Tuple of (responses_data, benchmark_data) in yQi format
    """
    
    # Load results
    results = []
    with open(results_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line.strip())
            results.append(result)
    
    # Sort results by custom_id to match original order
    results.sort(key=lambda x: int(x['custom_id'].split('-')[1]))
    
    # Process into yQi format
    responses_data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "system_prompt": system_prompt,
        "model": results[0]['response']['body']['model'] if results else "unknown",
        "processing_mode": "batch",
        "batch_processing": True,
        "batch_completed_at": datetime.now().isoformat(),
        "total_prompts": len(original_prompts),
        "responses": []
    }
    
    total_input_tokens = 0
    total_output_tokens = 0
    
    for i, result in enumerate(results):
        if result.get('error'):
            # Handle errors
            response_entry = {
                "prompt_number": i + 1,
                "prompt": original_prompts[i] if i < len(original_prompts) else "Unknown",
                "response": f"Error: {result['error']['message']}",
                "input_tokens": 0,
                "output_tokens": 0,
                "error": True
            }
        else:
            # Process successful response
            response_body = result['response']['body']
            usage = response_body.get('usage', {})
            
            response_entry = {
                "prompt_number": i + 1,
                "prompt": original_prompts[i] if i < len(original_prompts) else "Unknown",
                "response": response_body['choices'][0]['message']['content'],
                "input_tokens": usage.get('prompt_tokens', 0),
                "output_tokens": usage.get('completion_tokens', 0),
                "error": False
            }
            
            total_input_tokens += response_entry["input_tokens"]
            total_output_tokens += response_entry["output_tokens"]
        
        responses_data["responses"].append(response_entry)
    
    # Create benchmark data
    benchmark_data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "system_prompt": system_prompt,
        "model": responses_data["model"],
        "processing_mode": "batch",
        "batch_processing": True,
        "batch_completed_at": datetime.now().isoformat(),
        "total_prompts": len(original_prompts),
        "successful_responses": len([r for r in responses_data["responses"] if not r["error"]]),
        "failed_responses": len([r for r in responses_data["responses"] if r["error"]]),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens
    }
    
    return responses_data, benchmark_data


def process_prompts_batch(api_key: str, system_prompt: str, prompts: List[str], 
                         model: str = "gpt-4o-mini", temperature: float = 0.7, 
                         max_tokens: int = 500, status_text=None) -> Tuple[str, Dict]:
    """Process prompts using OpenAI Batch API.
    
    Args:
        api_key: OpenAI API key
        system_prompt: System prompt to use
        prompts: List of prompts to process
        model: OpenAI model to use
        temperature: Temperature setting
        max_tokens: Maximum tokens per response
        status_text: Streamlit status text object for updates
        
    Returns:
        Tuple of (batch_job_id, batch_info) for monitoring
    """
    import uuid
    from openai import OpenAI
    
    # Generate unique run ID
    run_id = str(uuid.uuid4())[:8]
    
    if status_text:
        status_text.text("Creating batch requests...")
    
    # Create batch requests
    batch_requests = create_batch_requests(
        system_prompt=system_prompt,
        prompts=prompts,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    if status_text:
        status_text.text("Saving batch file...")
    
    # Save batch file
    batch_file_path = save_batch_file(batch_requests, run_id)
    
    if status_text:
        status_text.text("Submitting batch job to OpenAI...")
    
    # Submit batch job
    client = OpenAI(api_key=api_key)
    batch_job_id = submit_batch_job(client, batch_file_path)
    
    # Store batch info for monitoring
    batch_info = {
        "batch_job_id": batch_job_id,
        "run_id": run_id,
        "system_prompt": system_prompt,
        "prompts": prompts,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "batch_file_path": batch_file_path,
        "submitted_at": datetime.now().isoformat(),
        "status": "submitted"
    }
    
    if status_text:
        status_text.text(f"Batch job submitted! Job ID: {batch_job_id}")
    
    logging.info(f"Batch processing initiated. Job ID: {batch_job_id}, Run ID: {run_id}")
    
    return batch_job_id, batch_info
    
