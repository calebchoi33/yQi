"""Test script for Tag RAG Proper system."""

import os
import logging
import json
from pathlib import Path

from database import setup_database, get_section_columns
from ingestion import ingest_all_sections
from query_engine import query, multi_family_query
from embeddings import TAG_FAMILIES

# Setup logging
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
    
    # Test queries for each tag family (complete set)
    test_queries = {
        "formulas": [
            "大陷胸湯",
            "Major Sinking into the Chest Decoction"
        ],
        "syndromes": [
            "太陽證",
            "Taiyang syndrome"
        ],
        "treatments": [
            "滋陰",
            "nourish yin"
        ],
        "pathogens": [
            "風寒",
            "wind-cold"
        ],
        "organs": [
            "心",
            "heart"
        ],
        "herbs": [
            "有毒的天然藥物",
            "toxic natural medicinals"
        ],
        "symptoms": [
            "發熱",
            "fever"
        ],
        "pulses": [
            "浮脈",
            "floating pulse"
        ],
        "acupoints": [
            "風池",
            "Fengchi"
        ],
        "meridians": [
            "太陽經",
            "Taiyang meridian"
        ],
        "elements": [
            "水",
            "water element"
        ],
        "tongues": [
            "舌苔",
            "tongue coating"
        ]
    }
    
    for family, queries in test_queries.items():
        print(f"\n--- {family.upper()} Family Tests ---")
        
        for query_text in queries:
            print(f"\nQuery: '{query_text}'")
            results = query(query_text, family, k=3, api_key=api_key)
            
            if results:
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. Book: {result['book_name']}")
                    print(f"     Chapter {result['chapter_index']}, Section {result['section_index']}")
                    if result['similarity_score']:
                        print(f"     Similarity: {result['similarity_score']:.3f}")
            else:
                print("  No results found")
    
    # Test multi-family query
    print(f"\n--- Multi-Family Query Test ---")
    query_text = "大陷胸湯 with heart symptoms"
    multi_results = multi_family_query(query_text, k=2, api_key=api_key)
    
    print(f"Query: '{query_text}'")
    for family, family_results in multi_results.items():
        print(f"\n{family.upper()} results ({len(family_results)}):")
        for result in family_results:
            print(f"  - Book: {result['book_name']}, Chapter: {result['chapter_index']}, Section: {result['section_index']}")
    
    return True

def test_database_structure():
    """Test database structure and content."""
    print("\n=== Testing Database Structure ===")
    
    import psycopg2
    conn = setup_database()
    
    # Check table exists and get columns
    columns = get_section_columns(conn)
    print(f"✓ Sections table columns: {columns}")
    
    # Check sections count
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM sections")
        count = cur.fetchone()[0]
        print(f"✓ Sections in database: {count}")
        
        # Check vector columns have data
        for family in TAG_FAMILIES.keys():
            vector_column = TAG_FAMILIES[family]["vector_column"]
            cur.execute(f"SELECT COUNT(*) FROM sections WHERE {vector_column} IS NOT NULL")
            vector_count = cur.fetchone()[0]
            print(f"✓ {family} vectors: {vector_count}")
        
        # Sample some data
        cur.execute("SELECT book_name, chapter_index, section_index FROM sections LIMIT 3")
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
    
    # Note: PostgreSQL database will be reused/reset automatically
    print("Using PostgreSQL database for testing")
    
    # Run tests
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
    
    # Summary
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
