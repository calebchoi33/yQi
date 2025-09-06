#!/usr/bin/env python3
"""Command-line evaluation runner for RAG system."""

import json
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import time

# Add models directory to path
sys.path.append(str(Path(__file__).parent.parent / 'models'))

from rag_system import RAGSystem
from document_processor import DocumentProcessor

# Load environment variables
from dotenv import load_dotenv
load_dotenv(str(Path(__file__).parent.parent / 'evaluation' / '.env'))

def load_config(config_path: str) -> Dict[str, Any]:
    """Load evaluation configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_patient_cases(cases_path: str) -> List[str]:
    """Load patient cases from JSON file."""
    with open(cases_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get('cases', [])

def create_evaluation_directories():
    """Create evaluation directory structure."""
    base_dir = Path("evaluation")
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Create directories
    responses_dir = base_dir / "responses" / today
    benchmarks_dir = base_dir / "benchmarks" / today
    
    responses_dir.mkdir(parents=True, exist_ok=True)
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    
    return responses_dir, benchmarks_dir

def run_rag_evaluation(config: Dict[str, Any], patient_cases: List[str], output_path: str):
    """Run RAG evaluation on patient cases."""
    
    start_time = time.time()
    
    # Initialize RAG system
    vector_db_path = config.get('vector_db_path', 'models/vector_db.pkl')
    rag_system = RAGSystem(vector_db_path=vector_db_path)
    
    # Check if vector database exists, if not build it
    if not rag_system.load_database():
        print("Building vector database from documents...")
        docs_dir = config.get('docs_directory', 'docs')
        processor = DocumentProcessor(docs_dir)
        chunks = processor.process_documents()
        
        if not chunks:
            print(f"❌ No documents found in {docs_dir}")
            return
        
        print(f"Processing {len(chunks)} chunks...")
        added = rag_system.build_database_from_chunks(chunks)
        print(f"Added {added} chunks to vector database")
        
        # Save database for future use
        rag_system.save_database()
    else:
        db_info = rag_system.get_database_info()
        print(f"Loaded existing database with {db_info['num_chunks']} chunks")
    
    # Create evaluation directories
    responses_dir, benchmarks_dir = create_evaluation_directories()
    
    # Generate run ID
    run_id = f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Prepare responses data (compatible with existing format)
    responses_data = {
        'run_id': run_id,
        'model': 'RAG_Local',
        'system_prompt': config.get('system_prompt', ''),
        'timestamp': datetime.now().isoformat(),
        'total_prompts': len(patient_cases),
        'responses': []
    }
    
    # Prepare benchmark data
    benchmark_data = {
        'run_id': run_id,
        'model': 'RAG_Local',
        'system_prompt': config.get('system_prompt', ''),
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'total_cases': len(patient_cases),
        'responses': [],
        'performance_metrics': {}
    }
    
    system_prompt = config.get('system_prompt', '')
    
    print(f"\nRunning evaluation on {len(patient_cases)} cases...")
    
    for i, case in enumerate(patient_cases, 1):
        print(f"Processing case {i}/{len(patient_cases)}...")
        
        case_start_time = time.time()
        
        try:
            # Query RAG system
            result = rag_system.query(case, top_n=config.get('top_n_chunks', 3), system_prompt=system_prompt)
            
            case_end_time = time.time()
            processing_time = case_end_time - case_start_time
            
            # Format response for responses file (compatible with existing format)
            response_entry = {
                'prompt': case,
                'response': result['response'],
                'model': 'RAG_Local',
                'timestamp': datetime.now().isoformat(),
                'prompt_number': i,
                'retrieved_chunks': result['retrieved_chunks'],
                'num_chunks_used': result['num_chunks_used'],
                'processing_time': processing_time
            }
            
            responses_data['responses'].append(response_entry)
            benchmark_data['responses'].append(response_entry)
            
        except Exception as e:
            print(f"❌ Error processing case {i}: {e}")
            case_end_time = time.time()
            processing_time = case_end_time - case_start_time
            
            error_entry = {
                'prompt': case,
                'response': f"Error: {e}",
                'model': 'RAG_Local',
                'timestamp': datetime.now().isoformat(),
                'prompt_number': i,
                'error': str(e),
                'processing_time': processing_time
            }
            
            responses_data['responses'].append(error_entry)
            benchmark_data['responses'].append(error_entry)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Add performance metrics
    benchmark_data['performance_metrics'] = {
        'total_processing_time': total_time,
        'average_time_per_case': total_time / len(patient_cases),
        'successful_cases': len([r for r in benchmark_data['responses'] if 'error' not in r]),
        'failed_cases': len([r for r in benchmark_data['responses'] if 'error' in r]),
        'database_chunks': rag_system.get_database_info()['num_chunks']
    }
    
    # Save responses file (compatible with existing evaluation platform)
    responses_filename = f"responses_{run_id}.json"
    responses_path = responses_dir / responses_filename
    
    with open(responses_path, 'w', encoding='utf-8') as f:
        json.dump(responses_data, f, ensure_ascii=False, indent=2)
    
    # Save benchmark file
    benchmark_filename = f"benchmark_{run_id}.json"
    benchmark_path = benchmarks_dir / benchmark_filename
    
    with open(benchmark_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
    
    # Also save to specified output path for backward compatibility
    if output_path:
        legacy_output = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'llm_responses': [
                {
                    'case_number': i+1,
                    'prompt': r['prompt'],
                    'response': r['response'],
                    'model': r['model'],
                    'timestamp': r['timestamp']
                }
                for i, r in enumerate(responses_data['responses'])
            ],
            'retrieved_texts': [
                {
                    'case_number': i+1,
                    'prompt': r['prompt'],
                    'retrieved_chunks': r.get('retrieved_chunks', []),
                    'num_chunks_used': r.get('num_chunks_used', 0)
                }
                for i, r in enumerate(responses_data['responses'])
            ],
            'input_config': {
                'use_rag': config.get('use_rag', True),
                'system_prompt': config.get('system_prompt', ''),
                'model': config.get('model', 'gpt-4o-mini')
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(legacy_output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Evaluation complete!")
    print(f"📁 Responses saved to: {responses_path}")
    print(f"📊 Benchmark saved to: {benchmark_path}")
    if output_path:
        print(f"📄 Legacy output saved to: {output_path}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"📈 Success rate: {benchmark_data['performance_metrics']['successful_cases']}/{len(patient_cases)}")
    print(f"🔍 Database chunks used: {benchmark_data['performance_metrics']['database_chunks']}")

def main():
    parser = argparse.ArgumentParser(description='Run RAG evaluation on patient cases')
    parser.add_argument('--config', required=True, help='Path to eval_config.json')
    parser.add_argument('--input', required=True, help='Path to patient_cases.json')
    parser.add_argument('--output', required=True, help='Path to output results JSON')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Load configuration and cases
    try:
        config = load_config(args.config)
        patient_cases = load_patient_cases(args.input)
        
        print(f"Loaded config: {args.config}")
        print(f"Loaded {len(patient_cases)} patient cases from: {args.input}")
        print(f"Output will be saved to: {args.output}")
        
        # Run evaluation
        run_rag_evaluation(config, patient_cases, args.output)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
