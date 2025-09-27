"""Query engine for Tag RAG system."""

import logging
import os
import psycopg2
from typing import List, Dict, Any, Optional
import numpy as np

from database import setup_database, search_by_tag_family
from embeddings import get_embedding
from embeddings import DEFAULT_TOP_K, TAG_FAMILIES

logger = logging.getLogger(__name__)

def query(query_text: str, tag_family: str, k: int = DEFAULT_TOP_K, api_key: str = None) -> List[Dict[str, Any]]:
    """
    Query the database for similar sections by tag family.
    
    Args:
        query_text: The text to search for
        tag_family: The tag family to search in (symptoms, organs, formulas)
        k: Number of results to return
        api_key: OpenAI API key
        
    Returns:
        List of matching sections with metadata
    """
    if tag_family not in TAG_FAMILIES:
        raise ValueError(f"Unknown tag family: {tag_family}. Available: {list(TAG_FAMILIES.keys())}")
    
    # Get query embedding
    query_embedding = get_embedding(query_text, api_key)
    if query_embedding is None:
        logger.error("Failed to generate embedding for query")
        return []
    
    # Connect to database
    conn = setup_database()
    
    # Search database
    results = search_by_tag_family(conn, query_embedding, tag_family, k)
    
    # Format results
    formatted_results = []
    for result in results:
        formatted_result = {
            "book_name": result.get("book_name", ""),
            "chapter_index": result.get("chapter_index", ""),
            "section_index": result.get("section_index", ""),
            "page_index": result.get("page_index"),
            "tag_family": tag_family,
            "similarity_score": result.get("similarity", 0.0)
        }
        formatted_results.append(formatted_result)
    
    conn.close()
    return formatted_results

def multi_family_query(query_text: str, families: List[str] = None, k: int = DEFAULT_TOP_K, api_key: str = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Query multiple tag families and return combined results.
    
    Args:
        query_text: The text to search for
        families: List of tag families to search (default: all)
        k: Number of results per family
        api_key: OpenAI API key
        
    Returns:
        Dictionary with results per family
    """
    if families is None:
        families = list(TAG_FAMILIES.keys())
    
    results = {}
    for family in families:
        family_results = query(query_text, family, k, api_key)
        results[family] = family_results
    
    return results

def main():
    """Example usage of the query engine."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is required")
        return
    
    # Example queries
    
    # Single family query
    print("=== Formula Query ===")
    results = query("大陷胸湯", "formulas", k=3, api_key=api_key)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['book_name']} Chapter {result['chapter_index']} - Section {result['section_index']}")
        print()
    
    # Multi-family query
    print("=== Multi-family Query ===")
    multi_results = multi_family_query("大陷胸湯 nourish yin", k=2, api_key=api_key)
    for family, family_results in multi_results.items():
        print(f"\n{family.upper()} Results:")
        for result in family_results:
            print(f"  - {result['book_name']} Chapter {result['chapter_index']} Section {result['section_index']}")

if __name__ == "__main__":
    main()
