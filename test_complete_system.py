#!/usr/bin/env python3
"""
Comprehensive test script for the yQi TCM RAG system.
Tests all components: structured RAG, vector database, and tagging integration.
"""

import os
import sys
import json
from pathlib import Path

# Add paths for imports
sys.path.append('rag_model/structured_rag')
sys.path.append('rag_model/vector_rag')

from rag_model.structured_rag.structured_rag_system import StructuredRAGSystem

def test_database_content():
    """Test the database content and structure."""
    print("🔍 Testing Database Content")
    print("=" * 50)
    
    # Initialize system
    config_path = "rag_model/structured_rag/config.json"
    db_path = "rag_model/data/vector_dbs/structured_vector_db.pkl"
    
    rag_system = StructuredRAGSystem(config_path, db_path)
    rag_system.load_database()
    
    # Get database info
    info = rag_system.get_database_info()
    print(f"📊 Total records: {info['total_records']}")
    print(f"📚 Books: {info.get('books', 'N/A')}")
    print(f"🔍 Search types: {', '.join(rag_system.get_available_search_types())}")
    
    # Test sample searches
    test_cases = [
        {
            "query": "发热恶寒头痛",
            "search_type": "multi_vector",
            "description": "Multi-vector search for fever and chills"
        },
        {
            "query": "桂枝汤",
            "search_type": "full_content",
            "description": "Full content search for Gui Zhi Tang"
        },
        {
            "query": "太阳病",
            "search_type": "patterns_vector",
            "description": "Pattern search for Taiyang disease"
        }
    ]
    
    print("\n🧪 Testing Search Functions:")
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['description']}")
        try:
            result = rag_system.query(
                query=test['query'],
                search_type=test['search_type'],
                use_tag_expansion=True
            )
            
            chunks_used = result.get('search_results_used', len(result.get('retrieved_chunks', [])))
            print(f"   ✅ Found {chunks_used} relevant chunks")
            
            if result.get('chinese_response'):
                print(f"   🇨🇳 Chinese: {result['chinese_response'][:100]}...")
            if result.get('english_response'):
                print(f"   🇺🇸 English: {result['english_response'][:100]}...")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return rag_system

def test_tagging_integration():
    """Test the tagging system integration."""
    print("\n\n🏷️  Testing Tagging Integration")
    print("=" * 50)
    
    tags_file = "rag_model/data/tags.json"
    if os.path.exists(tags_file):
        with open(tags_file, 'r', encoding='utf-8') as f:
            tags_data = json.load(f)
        
        print(f"📊 Total tags: {len(tags_data)}")
        
        # Show sample tags
        sample_tags = list(tags_data.items())[:5]
        print("\n🔖 Sample tags:")
        for key, value in sample_tags:
            if isinstance(value, dict):
                print(f"   {key}: {value.get('chinese', 'N/A')} | {value.get('english', 'N/A')}")
            else:
                print(f"   {key}: {value}")
    else:
        print("❌ Tags file not found")

def run_interactive_query(rag_system):
    """Run an interactive query session."""
    print("\n\n💬 Interactive Query Session")
    print("=" * 50)
    print("Enter TCM clinical cases or questions (type 'quit' to exit):")
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            # Use multi_vector search for comprehensive results
            result = rag_system.query(
                query=query,
                search_type="multi_vector",
                use_tag_expansion=True
            )
            
            print(f"\n📊 Found {result.get('search_results_used', 0)} relevant chunks")
            print("\n🇨🇳 Chinese Response:")
            print(result.get('chinese_response', 'No response'))
            print("\n🇺🇸 English Response:")
            print(result.get('english_response', 'No response'))
            print("-" * 40)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main test function."""
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    print("🧬 yQi TCM RAG System - Comprehensive Test")
    print("=" * 60)
    
    # Test database
    rag_system = test_database_content()
    
    # Test tagging
    test_tagging_integration()
    
    # Interactive session
    try:
        run_interactive_query(rag_system)
    except KeyboardInterrupt:
        pass
    
    print("\n✅ Testing complete!")
    print("\n💡 To run the Streamlit UI:")
    print("   cd evaluation && streamlit run app.py")

if __name__ == "__main__":
    main()
