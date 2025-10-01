"""Test script for Tag RAG Proper system (SQLite + sqlite-vec)."""

import os
import logging
import json
from pathlib import Path

from database import setup_database, get_section_columns
from ingestion import ingest_all_sections
from query_engine import query, multi_key_query
from embeddings import TAG_KEYS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ingestion():
    """Test the data ingestion pipeline."""
    print("=== Testing Data Ingestion ===")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is required")
        return False
    
    count = ingest_all_sections(api_key)
    print(f"✓ Successfully ingested {count} sections")
    return True

def test_queries():
    """Test various query scenarios."""
    print("\n=== Testing Query Engine ===")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is required")
        return False
    
    print(f"Tag keys: {TAG_KEYS}")
    
    test_queries = {
        "formulas": "大陷胸湯",
        "syndromes": "太陽證",
        "treatments": "滋陰",
        "symptoms": "發熱",
        "organs": "心",
    }
    
    for tag_key, query_text in test_queries.items():
        print(f"\n--- {tag_key.upper()} Tests ---")
        print(f"Query: '{query_text}'")
        results = query(query_text, tag_key, k=3, api_key=api_key)
        
        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Book: {result['book_name']}")
                print(f"     Chapter {result['chapter_index']}, Section {result['section_index']}")
                if result['similarity_score']:
                    print(f"     Similarity: {result['similarity_score']:.3f}")
        else:
            print("  No results found")
    
    print(f"\n--- Multi-Key Query Test ---")
    query_text = "大陷胸湯 with heart symptoms"
    test_keys = ["formulas", "symptoms", "organs"]
    multi_results = multi_key_query(query_text, test_keys, k=2, api_key=api_key)
    
    print(f"Query: '{query_text}'")
    for tag_key, key_results in multi_results.items():
        print(f"\n{tag_key.upper()} results ({len(key_results)}):")
        for result in key_results:
            print(f"  - Book: {result['book_name']}, Chapter: {result['chapter_index']}, Section: {result['section_index']}")
    
    return True

def test_database_structure():
    """Test database structure and content (SQLite + sqlite-vec)."""
    print("\n=== Testing Database Structure ===")
    conn = setup_database()
    cur = conn.cursor()

    columns = get_section_columns(conn)
    print(f"✓ vec_joined table columns: {columns}")

    cur.execute("SELECT COUNT(*) FROM vec_joined")
    count = cur.fetchone()[0]
    print(f"✓ Sections in database: {count}")

    print(f"\nDiscovered vector columns:")
    meta_cols = {"book_name", "chapter_index", "section_index", "page_index"}
    vec_cols = [col for col in columns if col not in meta_cols]
    for vec_col in vec_cols:
        print(f"  • {vec_col}")

    cur.execute("SELECT book_name, chapter_index, section_index FROM vec_joined LIMIT 3")
    samples = cur.fetchall()

    print(f"\n--- Sample Data ---")
    for i, sample in enumerate(samples, 1):
        book_name, chapter_index, section_id = sample
        print(f"{i}. {book_name} Ch{chapter_index} - Section {section_id}")

    conn.close()
    return True

def main():
    """Run all tests."""
    print("Starting Tag RAG Proper System Tests")
    print("=" * 50)
    print("Using SQLite + sqlite-vec database for testing")
    
    tests = [
        ("Data Ingestion", test_ingestion),
        ("Database Structure", test_database_structure),
        ("Query Engine", test_queries)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The Tag RAG Proper system is working correctly.")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please check the logs above.")

if __name__ == "__main__":
    main()
