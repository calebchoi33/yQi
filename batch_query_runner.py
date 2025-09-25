#!/usr/bin/env python3
"""
Automated batch query runner for yQi TCM RAG system.
Processes all queries from patient_cases.json and stores bilingual responses.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add structured_rag to path
sys.path.append('rag_model/structured_rag')
sys.path.append('rag_model/vector_rag')

from rag_model.structured_rag.structured_rag_system import StructuredRAGSystem

def load_patient_cases(cases_file="rag_model/config/patient_cases.json"):
    """Load patient cases from JSON file."""
    try:
        with open(cases_file, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        print(f"📋 Loaded {len(cases)} patient cases from {cases_file}")
        return cases
    except Exception as e:
        print(f"❌ Error loading patient cases: {e}")
        return []

def save_batch_results(results, output_dir="rag_model/output"):
    """Save batch query results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"batch_rag_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Create comprehensive results structure
    batch_data = {
        "run_id": f"batch_rag_{timestamp}",
        "timestamp": datetime.now().isoformat(),
        "model": "Structured RAG System",
        "total_queries": len(results),
        "successful_queries": len([r for r in results if not r.get('error')]),
        "failed_queries": len([r for r in results if r.get('error')]),
        "results": results
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(batch_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Results saved to: {filepath}")
    return filepath

def run_batch_queries():
    """Main function to run batch queries."""
    print("🧬 yQi TCM RAG System - Batch Query Runner")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Load patient cases
    patient_cases = load_patient_cases()
    if not patient_cases:
        print("❌ No patient cases found")
        sys.exit(1)
    
    # Initialize RAG system
    print("\n🔧 Initializing Structured RAG System...")
    config_path = "rag_model/structured_rag/config.json"
    db_path = "rag_model/data/vector_dbs/structured_vector_db.pkl"
    
    rag_system = StructuredRAGSystem(config_path, db_path)
    rag_system.load_database()
    
    # Show system info
    db_info = rag_system.get_database_info()
    print(f"✅ Database loaded: {db_info['total_records']} records")
    search_types = rag_system.get_available_search_types()
    print(f"🔍 Available search types: {', '.join(search_types)}")
    
    # Process all queries
    print(f"\n🚀 Processing {len(patient_cases)} patient cases...")
    results = []
    
    for i, case in enumerate(patient_cases, 1):
        print(f"\n📋 Case {i}/{len(patient_cases)}")
        print(f"Query: {case[:100]}...")
        
        try:
            # Query with multi-vector search and tag expansion
            result = rag_system.query(
                query=case,
                search_type="multi_vector",
                use_tag_expansion=True
            )
            
            # Store comprehensive result
            case_result = {
                "case_id": i,
                "query": case,
                "chinese_response": result.get('chinese_response', ''),
                "english_response": result.get('english_response', ''),
                "combined_response": result.get('combined_response', ''),
                "search_results_used": result.get('search_results_used', 0),
                "retrieved_chunks": result.get('retrieved_chunks', []),
                "model": result.get('model', 'gpt-4o-mini'),
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            results.append(case_result)
            
            # Show brief summary
            chunks_used = result.get('search_results_used', 0)
            print(f"✅ Success - {chunks_used} chunks used")
            print(f"🇨🇳 Chinese: {result.get('chinese_response', '')[:80]}...")
            print(f"🇺🇸 English: {result.get('english_response', '')[:80]}...")
            
        except Exception as e:
            print(f"❌ Error processing case {i}: {e}")
            
            # Store error result
            error_result = {
                "case_id": i,
                "query": case,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            results.append(error_result)
    
    # Save results
    print(f"\n💾 Saving batch results...")
    output_file = save_batch_results(results)
    
    # Summary
    successful = len([r for r in results if r.get('success')])
    failed = len([r for r in results if not r.get('success')])
    
    print(f"\n📊 Batch Processing Summary:")
    print(f"✅ Successful queries: {successful}")
    print(f"❌ Failed queries: {failed}")
    print(f"📁 Results saved to: {output_file}")
    
    # Show sample result
    if successful > 0:
        sample = next(r for r in results if r.get('success'))
        print(f"\n📝 Sample Result (Case {sample['case_id']}):")
        print(f"🔍 Query: {sample['query'][:100]}...")
        print(f"🇨🇳 Chinese: {sample['chinese_response'][:150]}...")
        print(f"🇺🇸 English: {sample['english_response'][:150]}...")
        print(f"📊 Chunks used: {sample['search_results_used']}")
    
    print(f"\n🎉 Batch processing complete!")
    return output_file

if __name__ == "__main__":
    run_batch_queries()
