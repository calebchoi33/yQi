#!/usr/bin/env python3
"""
Diagnosis Router - Routes diagnosis requests to different workflows.

This module serves as the main entry point for TCM diagnosis inference,
routing requests to appropriate workflows based on configuration and requirements.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass

from workflows.chunk_rag.vector_rag.rag_system import RAGSystem
from workflows.tag_rag.structured_rag.structured_rag_system import StructuredRAGSystem

@dataclass
class DiagnosisOutput:
    """Output class for diagnosis results."""
    diagnosis: str
    metadata: Dict[str, Any]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    logger.info(f"Loaded configuration from {config_path} with {len(config['prompts'])} prompts")
    return config




def process_all_prompts(config: Dict[str, Any]) -> List[DiagnosisOutput]:
    """Process all prompts from the configuration file."""
    results = []
    prompts = config["prompts"]
    
    logger.info(f"Processing {len(prompts)} prompts using {config['workflow']} workflow")
    
    for i, prompt_data in enumerate(prompts, 1):
        result = give_diagnosis(config, prompt_data["content"])
        
        # Add prompt metadata to the result
        result.metadata.update({
            "prompt_id": prompt_data.get("id", f"prompt_{i}"),
            "prompt_description": prompt_data.get("description", ""),
            "prompt_index": i
        })
        
        results.append(result)
        logger.info(f"Processed prompt {i}/{len(prompts)}: {prompt_data.get('id', f'prompt_{i}')}")
    
    return results


def give_diagnosis(config: Dict[str, Any], patient_case: str) -> DiagnosisOutput:
    """Process a single diagnosis request."""
    workflow_type = config["workflow"]
    
    # Execute diagnosis based on workflow type
    start_time = time.time()
    
    if workflow_type == "no_rag":
        # Direct LLM approach
        from openai import OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        
        client = OpenAI(api_key=api_key)
        system_prompt = """You are an expert Traditional Chinese Medicine (TCM) practitioner. 
        Analyze the patient case and provide a comprehensive TCM diagnosis including:
        1. Syndrome differentiation (辨證)
        2. Treatment principles (治則)
        3. Recommended herbal formula with specific herbs and dosages
        4. Lifestyle recommendations
        
        Respond in both Chinese and English where appropriate."""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": patient_case}
            ],
            temperature=0.3
        )
        
        processing_time = time.time() - start_time
        return DiagnosisOutput(
            diagnosis=response.choices[0].message.content,
            metadata={
                "workflow": "no_rag",
                "model": "gpt-4o-mini",
                "processing_time": processing_time
            }
        )
    
    elif workflow_type == "chunk_rag":
        # Chunk-based RAG approach
        rag_system = RAGSystem()
        retrieved_chunks = rag_system.retrieve(patient_case)
        diagnosis = rag_system.generate_response(patient_case, retrieved_chunks)
        
        processing_time = time.time() - start_time
        return DiagnosisOutput(
            diagnosis=diagnosis,
            metadata={
                "workflow": "chunk_rag",
                "retrieved_chunks": len(retrieved_chunks),
                "processing_time": processing_time
            }
        )
    
    elif workflow_type == "tag_rag":
        # Structured tag-based RAG approach
        # Initialize structured RAG system (would need proper config paths)
        structured_rag = StructuredRAGSystem(
            config_path="config/structured_config.json",
            db_path="data/structured_db.pkl"
        )
        
        search_results = structured_rag.search(patient_case)
        response = structured_rag.generate_response(patient_case, search_results)
        
        processing_time = time.time() - start_time
        return DiagnosisOutput(
            diagnosis=response.get("response", "No response generated"),
            metadata={
                "workflow": "tag_rag",
                "search_results": len(search_results),
                "processing_time": processing_time
            }
        )
    
    else:
        return DiagnosisOutput(
            diagnosis="Error: Unknown workflow type",
            metadata={"workflow": workflow_type, "error": "unknown_workflow"}
        )


def main():
    """CLI interface for diagnosis router."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TCM Diagnosis Inference Router")
    parser.add_argument("--config", "-f", required=True, help="Path to configuration file (required)")
    parser.add_argument("--case", "-c", help="Single patient case description (optional, overrides config prompts)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", help="Output file to save results (JSON format)")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    if args.case:
        # Process single case
        response = give_diagnosis(config, args.case)
        
        print(f"\n=== TCM Diagnosis ({response.metadata.get('workflow', 'unknown')}) ===")
        print(f"Case: {args.case[:100]}...")
        print(f"Diagnosis: {response.diagnosis}")
        print(f"Processing Time: {response.metadata.get('processing_time', 0):.2f}s")
        
        if args.verbose and response.metadata:
            print(f"\nMetadata: {json.dumps(response.metadata, indent=2)}")
            
        results = [response]
    else:
        # Process all prompts from config
        results = process_all_prompts(config)
        
        print(f"\n=== TCM Diagnosis Results ({config['workflow']}) ===")
        print(f"Processed {len(results)} prompts\n")
        
        for i, response in enumerate(results, 1):
            prompt_id = response.metadata.get('prompt_id', f'prompt_{i}')
            description = response.metadata.get('prompt_description', '')
            
            print(f"--- Case {i}: {prompt_id} ---")
            if description:
                print(f"Description: {description}")
            print(f"Diagnosis: {response.diagnosis}")
            print(f"Processing Time: {response.metadata.get('processing_time', 0):.2f}s")
            
            if args.verbose and response.metadata:
                print(f"Metadata: {json.dumps(response.metadata, indent=2)}")
            
            print()
    
    # Save results to file if requested
    if args.output:
        # Convert DiagnosisOutput objects to dictionaries for JSON serialization
        results_dict = []
        for result in results:
            results_dict.append({
                "diagnosis": result.diagnosis,
                "metadata": result.metadata
            })
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
