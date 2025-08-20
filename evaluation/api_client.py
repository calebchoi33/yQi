"""OpenAI API client for ChatGPT interactions with benchmarking.

This module handles all interactions with the OpenAI API, including sending prompts,
processing responses, and collecting detailed benchmarking metrics. It provides
real-time progress tracking through Streamlit's UI components.
"""

import time
import logging
from datetime import datetime
from typing import List, Dict, Tuple
import openai
from openai import OpenAI
import streamlit as st
from config import MODEL_NAME, MAX_TOKENS, TEMPERATURE, MAX_RETRIES, RETRY_DELAY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_client.log"),
        logging.StreamHandler()
    ]
)


def send_prompts_to_chatgpt(prompts: List[str], api_key: str) -> Tuple[Dict, Dict]:
    """Send a series of prompts to ChatGPT API and collect responses with benchmarking.
    
    This function processes each prompt sequentially, sending it to the OpenAI API,
    and collects both the response content and detailed performance metrics.
    It displays a progress bar and status updates in the Streamlit UI.
    
    Args:
        prompts: List of text prompts to send to the API
        api_key: OpenAI API key for authentication
        
    Returns:
        Tuple containing:
            - responses_data: Dictionary with all prompt-response pairs and metadata
            - benchmark_data: Dictionary with detailed performance metrics
    """
    client = OpenAI(api_key=api_key)
    
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
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, prompt in enumerate(prompts):
        prompt_start_time = time.time()
        retry_count = 0
        success = False
        
        while not success and retry_count <= MAX_RETRIES:
            try:
                status_text.text(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
                
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE
                )
                
                prompt_end_time = time.time()
                prompt_duration = prompt_end_time - prompt_start_time
                success = True
                
                # Log successful API call
                logging.info(f"Successfully processed prompt {i+1}/{len(prompts)}")
                
            except openai.RateLimitError as e:
                retry_count += 1
                if retry_count <= MAX_RETRIES:
                    wait_time = RETRY_DELAY * retry_count
                    logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds... (Attempt {retry_count}/{MAX_RETRIES})")
                    status_text.text(f"Rate limit exceeded. Retrying in {wait_time} seconds... (Attempt {retry_count}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                else:
                    raise
                    
            except openai.APIConnectionError as e:
                retry_count += 1
                if retry_count <= MAX_RETRIES:
                    wait_time = RETRY_DELAY * retry_count
                    logging.warning(f"Connection error. Retrying in {wait_time} seconds... (Attempt {retry_count}/{MAX_RETRIES})")
                    status_text.text(f"Connection error. Retrying in {wait_time} seconds... (Attempt {retry_count}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                else:
                    raise
            
            except Exception as e:
                # For other exceptions, don't retry
                logging.error(f"Error processing prompt {i+1}: {str(e)}")
                raise
        
        if success:
            
            response_data = {
                "prompt_number": i + 1,
                "prompt": prompt,
                "response": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens,
                "timestamp": datetime.now().isoformat()
            }
            
            responses_data["responses"].append(response_data)
            
            # Update benchmark data
            benchmark_data["prompt_processing_times"].append({
                "prompt_number": i + 1,
                "duration_seconds": prompt_duration,
                "tokens_used": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            })
            benchmark_data["total_tokens_used"] += response.usage.total_tokens
            benchmark_data["successful_prompts"] += 1
            
            progress_bar.progress((i + 1) / len(prompts))
            
        if not success:
            prompt_end_time = time.time()
            prompt_duration = prompt_end_time - prompt_start_time
            
            error_message = f"Failed to process prompt {i+1} after {MAX_RETRIES} retries"
            logging.error(error_message)
            st.error(error_message)
            
            response_data = {
                "prompt_number": i + 1,
                "prompt": prompt,
                "response": f"Error: Maximum retries exceeded",
                "tokens_used": 0,
                "timestamp": datetime.now().isoformat()
            }
            responses_data["responses"].append(response_data)
            
            # Update benchmark data for errors
            benchmark_data["prompt_processing_times"].append({
                "prompt_number": i + 1,
                "duration_seconds": prompt_duration,
                "tokens_used": 0,
                "error": "Maximum retries exceeded"
            })
            benchmark_data["failed_prompts"] += 1
            benchmark_data["errors"].append({
                "prompt_number": i + 1,
                "error": "Maximum retries exceeded",
                "timestamp": datetime.now().isoformat()
            })
    
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
    
    status_text.text("All prompts processed!")
    return responses_data, benchmark_data
