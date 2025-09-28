#!/usr/bin/env python3
"""
Test script for the structured RAG system.
Tests multi-vector search and bilingual response generation.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../evaluation/.env')

from structured_rag_system import StructuredRAGSystem

def test_structured_rag():
    """Test the structured RAG system with sample TCM queries."""
    
    # Initialize the structured RAG system
    config_path = "config.json"
    db_path = "../data/vector_dbs/structured_vector_db.pkl"
    
    print("🔬 Testing Structured RAG System")
    print("=" * 50)
    
    # Initialize system
    rag_system = StructuredRAGSystem(config_path, db_path)
    
    # Load the database
    print(f"📚 Loading database from: {db_path}")
    success = rag_system.load_database()
    
    if not success:
        print("❌ Failed to load database")
        return
    
    # Display database info
    info = rag_system.get_database_info()
    print(f"✅ Database loaded: {info['total_records']} records")
    search_types = rag_system.get_available_search_types()
    print(f"📊 Available search types: {', '.join(search_types)}")
    
    # Check if vectors exist
    if rag_system.structured_db.records:
        first_record = rag_system.structured_db.records[0]
        has_vectors = bool(first_record.full_content_vector)
        print(f"🔍 Vectors exist: {has_vectors}")
        
        if not has_vectors:
            print("❌ Database has no embeddings - rebuilding...")
            rebuild_with_embeddings(rag_system)
    
    # Test a simple search
    print("\n🔍 Testing search functionality...")
    test_query = "太陽病發熱惡寒"
    results = rag_system.search(test_query, search_type="multi_vector", top_k=3)
    print(f"📊 Search results for '{test_query}': {len(results)} chunks found")
    
    if results:
        print("✅ Search working properly")
        for i, result in enumerate(results[:2]):
            print(f"  Result {i+1}: {result['content'][:100]}...")
    else:
        print("❌ No search results - system not working")

def rebuild_with_embeddings(rag_system):
    """Rebuild database with proper embeddings."""
    print("🔧 Rebuilding database with embeddings...")
    
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return
    
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    
    # Set client and rebuild
    rag_system.structured_db.openai_client = client
    total = rag_system.structured_db.build_from_documents()
    
    if total > 0:
        rag_system.structured_db.save_database("../data/vector_dbs/structured_vector_db.pkl")
        print(f"✅ Rebuilt database with {total} records and embeddings")
        
        # Reload the database
        rag_system.load_database()
    else:
        print("❌ Failed to rebuild database")

if __name__ == "__main__":
    test_structured_rag()
    print()
    
    # Test queries
    test_queries = [
        {
            "query": "患者发热恶寒，头痛身痛，无汗，脉浮紧。请诊断并给出治疗方案。",
            "search_type": "multi_vector",
            "description": "Classic Taiyang syndrome case"
        },
        {
            "query": "What is the treatment for cold damage with fever and chills?",
            "search_type": "symptoms_vector",
            "description": "English symptom-focused query"
        },
        {
            "query": "桂枝汤的组成和功效是什么？",
            "search_type": "formulas_vector", 
            "description": "Formula-specific query"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"🔍 Test Query {i}: {test_case['description']}")
        print(f"Query: {test_case['query']}")
        print(f"Search Type: {test_case['search_type']}")
        print("-" * 40)
        
        try:
            # Generate response
            result = rag_system.query(
                query=test_case['query'],
                search_type=test_case['search_type'],
                use_tag_expansion=True
            )
            
            # Handle different result formats
            chunks_used = result.get('search_results_used', len(result.get('retrieved_chunks', [])))
            print(f"📊 Search Results: {chunks_used} chunks used")
            print(f"🤖 Model: {result.get('model', 'gpt-4o-mini')}")
            print()
            
            print("🇨🇳 Chinese Response:")
            print(result['chinese_response'])
            print()
            
            print("🇺🇸 English Response:")
            print(result['english_response'])
            print()
            
            # Show retrieved chunks
            if 'retrieved_chunks' in result:
                print("📚 Retrieved Content:")
                for j, chunk in enumerate(result['retrieved_chunks'][:2], 1):
                    print(f"  {j}. {chunk.get('book_name', 'Unknown')} - Chapter {chunk.get('chapter_index', '?')}")
                    print(f"     {chunk.get('content', '')[:100]}...")
                print()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print()
        
        print("=" * 50)
        print()

if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    test_structured_rag()
