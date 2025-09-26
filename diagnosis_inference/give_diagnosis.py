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

# Add workflows to path
sys.path.append(str(Path(__file__).parent / "workflows"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Workflow types
WORKFLOW_TYPES = ["no_rag", "chunk_rag", "tag_rag"]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    logger.info(f"Loaded configuration from {config_path} with {len(config['prompts'])} prompts")
    return config


def create_no_rag_workflow():
    """Create no-RAG workflow (direct LLM)."""
    def process_no_rag(request: Dict[str, Any]) -> Dict[str, Any]:
        """Process diagnosis using direct LLM inference."""
        # TODO: Implement direct OpenAI API call
        return {
            "diagnosis": f"[NO-RAG] Direct LLM diagnosis for: {request['patient_case'][:100]}...",
            "confidence_score": 0.7,
            "metadata": {"workflow": "no_rag", "method": "direct_llm"}
        }
    return process_no_rag


def create_chunk_rag_workflow():
    """Create chunk-based RAG workflow."""
    sys.path.append(str(Path(__file__).parent / "workflows" / "chunk_rag" / "vector_rag"))
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


def create_tag_rag_workflow():
    """Create tag-based structured RAG workflow."""
    sys.path.append(str(Path(__file__).parent / "workflows" / "tag_rag" / "structured_rag"))
    # from structured_rag_system import StructuredRAGSystem
    
    def process_tag_rag(request: Dict[str, Any]) -> Dict[str, Any]:
        """Process diagnosis using structured tag-based RAG."""
        # TODO: Implement structured RAG system integration
        return {
            "diagnosis": f"[TAG-RAG] Structured diagnosis for: {request['patient_case'][:100]}...",
            "confidence_score": 0.9,
            "retrieved_context": ["Structured context 1", "Structured context 2"],
            "metadata": {"workflow": "tag_rag", "method": "multi_vector_search"}
        }
    return process_tag_rag


def initialize_workflow(workflow_type: str):
    """Initialize and return a workflow function."""
    if workflow_type == "no_rag":
        return create_no_rag_workflow()
    elif workflow_type == "chunk_rag":
        return create_chunk_rag_workflow()
    elif workflow_type == "tag_rag":
        return create_tag_rag_workflow()
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
    workflow_type = config["workflow"]
    workflow_func = initialize_workflow(workflow_type)
    
    results = []
    prompts = config["prompts"]
    
    logger.info(f"Processing {len(prompts)} prompts using {workflow_type} workflow")
    
    for i, prompt_data in enumerate(prompts, 1):
        start_time = time.time()
        
        # Create request from prompt data
        request = create_diagnosis_request(prompt_data["content"], workflow_type)
        
        result = workflow_func(request)
        processing_time = time.time() - start_time
        
        response = create_diagnosis_response(
            diagnosis=result["diagnosis"],
            workflow_used=workflow_type,
            confidence_score=result.get("confidence_score"),
            retrieved_context=result.get("retrieved_context"),
            processing_time=processing_time,
            metadata={
                **result.get("metadata", {}),
                "prompt_id": prompt_data.get("id", f"prompt_{i}"),
                "prompt_description": prompt_data.get("description", ""),
                "prompt_index": i
            }
        )
        
        results.append(response)
        logger.info(f"Processed prompt {i}/{len(prompts)}: {prompt_data.get('id', f'prompt_{i}')}")
    
    return results


def give_diagnosis(config: Dict[str, Any], patient_case: str) -> Dict[str, Any]:
    """Process a single diagnosis request."""
    workflow_type = config["workflow"]
    workflow_func = initialize_workflow(workflow_type)
    
    # Create request
    request = create_diagnosis_request(patient_case, workflow_type)
    
    # Execute diagnosis
    start_time = time.time()
    result = workflow_func(request)
    processing_time = time.time() - start_time
    
    return create_diagnosis_response(
        diagnosis=result["diagnosis"],
        workflow_used=workflow_type,
        confidence_score=result.get("confidence_score"),
        retrieved_context=result.get("retrieved_context"),
        processing_time=processing_time,
        metadata=result.get("metadata", {})
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
        
        print(f"\n=== TCM Diagnosis ({response['workflow_used']}) ===")
        print(f"Case: {args.case[:100]}...")
        print(f"Diagnosis: {response['diagnosis']}")
        print(f"Confidence: {response['confidence_score']}")
        print(f"Processing Time: {response['processing_time']:.2f}s")
        
        if response['retrieved_context']:
            print(f"\nRetrieved Context:")
            for i, context in enumerate(response['retrieved_context'], 1):
                print(f"  {i}. {context}")
        
        if args.verbose and response['metadata']:
            print(f"\nMetadata: {json.dumps(response['metadata'], indent=2)}")
            
        results = [response]
    else:
        # Process all prompts from config
        results = process_all_prompts(config)
        
        print(f"\n=== TCM Diagnosis Results ({config['workflow']}) ===")
        print(f"Processed {len(results)} prompts\n")
        
        for i, response in enumerate(results, 1):
            prompt_id = response['metadata'].get('prompt_id', f'prompt_{i}')
            description = response['metadata'].get('prompt_description', '')
            
            print(f"--- Case {i}: {prompt_id} ---")
            if description:
                print(f"Description: {description}")
            print(f"Diagnosis: {response['diagnosis']}")
            print(f"Confidence: {response['confidence_score']}")
            print(f"Processing Time: {response['processing_time']:.2f}s")
            
            if args.verbose and response['retrieved_context']:
                print(f"Retrieved Context:")
                for j, context in enumerate(response['retrieved_context'], 1):
                    print(f"  {j}. {context}")
            
            print()
    
    # Save results to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
