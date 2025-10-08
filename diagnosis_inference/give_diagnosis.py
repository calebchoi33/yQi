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
import argparse
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from openai import OpenAI

# Simple import path for tag_rag_proper
sys.path.append(str(Path(__file__).parent / "workflows" / "tag_rag_proper"))
from query_engine import diagnose
from embeddings import TAG_KEYS, DEFAULT_TOP_K


@dataclass
class DiagnosisOutput:
    """Output format for diagnosis results."""
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




def create_chunk_rag_workflow():
    """Create chunk-based RAG workflow."""
    # from rag_system import RAGSystem
    
    def process_chunk_rag(request: Dict[str, Any]) -> Dict[str, Any]:
        """Process diagnosis using chunk-based RAG."""
        # TODO: Implement chunk RAG system integration
        return {
            "diagnosis": f"[CHUNK-RAG] Chunk-based diagnosis for: {request['patient_case'][:100]}...",
            "confidence_score": 0.8,
            "retrieved_context": ["Sample retrieved chunk 1", "Sample retrieved chunk 2"],
            "metadata": {"workflow": "chunk_rag", "method": "vector_similarity"}
        }
    return process_chunk_rag




def initialize_workflow(workflow_type: str):
    """Initialize and return a workflow function."""
    if workflow_type == "no_rag":
        return create_no_rag_workflow()
    elif workflow_type == "chunk_rag":
        return create_chunk_rag_workflow()
    else:
        return None


def create_diagnosis_request(patient_case: str, workflow_type: str = "chunk_rag") -> Dict[str, Any]:
    """Create a diagnosis request dictionary."""
    return {
        "patient_case": patient_case,
        "workflow_type": workflow_type,
        "language": "bilingual"
    }


def create_diagnosis_response(diagnosis: str, workflow_used: str, confidence_score: float = None, 
                            retrieved_context: List[str] = None, processing_time: float = None, 
                            metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a diagnosis response dictionary."""
    return {
        "diagnosis": diagnosis,
        "workflow_used": workflow_used,
        "confidence_score": confidence_score,
        "retrieved_context": retrieved_context or [],
        "processing_time": processing_time,
        "metadata": metadata or {}
    }


def process_all_prompts(config: Dict[str, Any]) -> List[Dict[str, Any]]:
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
            "prompt_index": i,
            "prompt_content": prompt_data.get("content", ""),
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
        # Tag-based RAG using tag_rag_proper system
        api_key = os.getenv('OPENAI_API_KEY')

        # Allow tag keys and k to be provided in config; default to full TAG_KEYS and DEFAULT_TOP_K
        config_tag_keys = config.get("tag_keys")
        tag_keys = config_tag_keys if isinstance(config_tag_keys, list) and config_tag_keys else list(TAG_KEYS)
        k = int(config.get("k", DEFAULT_TOP_K))

        # Use tag_rag_proper's diagnose function
        result = diagnose(
            patient_case=patient_case,
            tag_keys=tag_keys,
            k=k,
            api_key=api_key
        )

        # Compute counts from returned structure
        retrieved_ctx = result.get("retrieved_context", {})
        total_retrieved = sum(len(v) for v in retrieved_ctx.values()) if isinstance(retrieved_ctx, dict) else 0
        meta = result.get("metadata", {})

        processing_time = time.time() - start_time
        return DiagnosisOutput(
            diagnosis=result.get("diagnosis", "No diagnosis generated"),
            metadata={
                "workflow": "tag_rag",
                "retrieved_sections": total_retrieved,
                "tag_keys_used": meta.get("tag_keys_searched", tag_keys),
                "results_per_key": meta.get("results_per_key", k),
                "formatted_context": result.get("formatted_context", ""),
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
            # Exclude certain metadata fields from the saved JSON
            exclude_keys = {
                "tag_keys_used",
                "processing_time",
                "prompt_description",
                "prompt_id",
                "prompt_index",
                "retrieved_sections",
                "workflow",
                "results_per_key",
                "prompt_content",
            }
            metadata_clean = {k: v for k, v in (result.metadata or {}).items() if k not in exclude_keys}
            prompt_text = (result.metadata or {}).get("prompt_content", "")
            # Place 'prompt' before 'diagnosis' for readability
            results_dict.append({
                "prompt": prompt_text,
                "diagnosis": result.diagnosis,
                "metadata": metadata_clean
            })
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
