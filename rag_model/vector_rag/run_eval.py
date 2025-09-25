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
# Import models and evaluation modules
from document_processor import DocumentProcessor
from rag_system import RAGSystem
from enhanced_rag_system import EnhancedRAGSystem
from evaluation.core.config import EvaluationConfig
from evaluation.core.evaluator import Evaluator
from evaluation.core.prompts import PromptManager

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

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
    use_rag_mode = config.get('use_rag', 'True')
    rag_system = RAGSystem(vector_db_path=vector_db_path, use_rag=str(use_rag_mode))
    
    # Handle different RAG modes
    if use_rag_mode == "Mock":
        # Load mock data if available
        mock_data_path = config.get('mock_data_path', 'evaluation/mock_retrieval_data.json')
        if os.path.exists(mock_data_path):
            with open(mock_data_path, 'r', encoding='utf-8') as f:
                mock_data = json.load(f)
            rag_system.set_mock_retrieval_data(mock_data)
            print(f"📋 Loaded mock retrieval data from {mock_data_path}")
        else:
            print(f"⚠️  Mock mode enabled but no mock data found at {mock_data_path}")
    elif use_rag_mode == "False":
        print("🚫 RAG disabled - using LLM only")
    else:
        # Regular RAG mode - load or build vector database
        if not rag_system.load_database():
            print("Building vector database from documents...")
            docs_dir = config.get('docs_directory', '../data/documents')
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
                'chinese_response': result.get('chinese_response', ''),
                'english_response': result.get('english_response', ''),
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
    
    # Also save to specified output path with improved format
    if output_path:
        # Generate timestamped filename if output_path doesn't already have timestamp
        if 'semantic_rag_evaluation' in output_path and not any(char.isdigit() for char in output_path.split('/')[-1]):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_path = output_path.replace('.json', '')
            output_path = f"{base_path}_{timestamp}.json"
        
        # Consolidated format with responses and retrieved texts together
        consolidated_output = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'evaluation_cases': [
                {
                    'case_number': i+1,
                    'prompt': r['prompt'],
                    'response': r['response'],
                    'chinese_response': r.get('chinese_response', ''),
                    'english_response': r.get('english_response', ''),
                    'model': r['model'],
                    'timestamp': r['timestamp'],
                    'retrieved_chunks': r.get('retrieved_chunks', []),
                    'num_chunks_used': r.get('num_chunks_used', 0),
                    'processing_time': r.get('processing_time', 0)
                }
                for i, r in enumerate(responses_data['responses'])
            ],
            'summary': {
                'total_cases': len(responses_data['responses']),
                'successful_cases': len([r for r in responses_data['responses'] if 'error' not in r]),
                'failed_cases': len([r for r in responses_data['responses'] if 'error' in r]),
                'total_processing_time': sum(r.get('processing_time', 0) for r in responses_data['responses']),
                'average_processing_time': sum(r.get('processing_time', 0) for r in responses_data['responses']) / len(responses_data['responses']) if responses_data['responses'] else 0
            },
            'input_config': {
                'use_rag': config.get('use_rag', True),
                'system_prompt': config.get('system_prompt', ''),
                'model': config.get('model', 'gpt-4o-mini')
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(consolidated_output, f, ensure_ascii=False, indent=2)
        
        print(f"📄 Consolidated output saved to: {output_path}")
    
    print(f"✅ Evaluation complete!")
    print(f"📁 Responses saved to: {responses_path}")
    print(f"📊 Benchmark saved to: {benchmark_path}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"📈 Success rate: {benchmark_data['performance_metrics']['successful_cases']}/{len(patient_cases)}")
    print(f"🔍 Database chunks used: {benchmark_data['performance_metrics']['database_chunks']}")

def main():
    parser = argparse.ArgumentParser(description='Run RAG evaluation on patient cases')
    parser.add_argument('--config', required=True, help='Path to eval_config.json')
    parser.add_argument('--output', required=True, help='Path to output results JSON')
    parser.add_argument('--custom-prompts', help='Optional: Path to custom prompts JSON file (overrides default prompts.py)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    # Load configuration and prompts
    try:
        config = load_config(args.config)
        
        # Always use prompts from evaluation/prompts.py as default
        sys.path.append(str(Path(__file__).parent.parent.parent / 'evaluation'))
        from prompts import DEFAULT_PROMPTS
        
        if args.custom_prompts:
            if not os.path.exists(args.custom_prompts):
                print(f"❌ Custom prompts file not found: {args.custom_prompts}")
                sys.exit(1)
            patient_cases = load_patient_cases(args.custom_prompts)
            print(f"✅ Loaded {len(patient_cases)} custom prompts from: {args.custom_prompts}")
        else:
            patient_cases = DEFAULT_PROMPTS
            print(f"✅ Using {len(patient_cases)} prompts from evaluation/prompts.py")
        
        print(f"📋 Configuration loaded: {args.config}")
        print(f"📝 Total prompts to evaluate: {len(patient_cases)}")
        print(f"💾 Output will be saved to: {args.output}")
        
        # Run evaluation
        run_rag_evaluation(config, patient_cases, args.output)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
