"""OpenAI API client for ChatGPT interactions with benchmarking.

This module handles all interactions with the OpenAI API, including sending prompts,
processing responses, and collecting detailed benchmarking metrics. It provides
real-time progress tracking through Streamlit's UI components.

Implements asynchronous processing for improved performance when handling multiple prompts.
"""

import time
import logging
import asyncio
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
    
