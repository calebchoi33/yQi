"""Query engine for Tag RAG system (SQLite + sqlite-vec)."""

import logging
import os
from typing import List, Dict, Any

from database import setup_database, search_by_tag_key
from embeddings import get_embedding, DEFAULT_TOP_K, _get_client

logger = logging.getLogger(__name__)

def query(query_text: str, tag_key: str, k: int = DEFAULT_TOP_K, api_key: str = None, conn = None) -> List[Dict[str, Any]]:
    """Search nearest sections by a tag key.

    Args:
        query_text: Text to embed and search for.
        tag_key: Tag key to search (e.g., 'formulas', 'syndromes', 'symptoms').
        k: Max results to return.
        api_key: OpenAI API key.
        conn: Optional database connection to reuse. If None, creates and closes one.

    Returns:
        List of section metadata dicts with a similarity score.
    """
    query_embedding = get_embedding(query_text, api_key)
    if query_embedding is None:
        logger.error("Failed to generate embedding for query")
        return []

    should_close = False
    if conn is None:
        conn = setup_database()
        should_close = True

    results = search_by_tag_key(conn, query_embedding, tag_key, k)

    formatted_results = []
    for result in results:
        formatted_result = {
            "book_name": result.get("book_name", ""),
            "chapter_index": result.get("chapter_index", ""),
            "section_index": result.get("section_index", ""),
            "page_index": result.get("page_index"),
            "tag_key": tag_key,
            "similarity_score": result.get("similarity", 0.0),
        }
        formatted_results.append(formatted_result)

    if should_close:
        conn.close()
    return formatted_results

def multi_key_query(query_text: str, tag_keys: List[str], k: int = DEFAULT_TOP_K, api_key: str = None) -> Dict[str, List[Dict[str, Any]]]:
    """Query multiple tag keys and return results grouped by key.

    Args:
        query_text: Text to embed and search for.
        tag_keys: List of tag keys to query (e.g., ['formulas', 'symptoms']).
        k: Max results to return per tag key.
        api_key: OpenAI API key.

    Returns:
        Dict mapping tag keys to lists of section metadata dicts with similarity scores.
    """
    query_embedding = get_embedding(query_text, api_key)
    if query_embedding is None:
        logger.error("Failed to generate embedding for query")
        return {}

    conn = setup_database()
    results = {}
    
    for tag_key in tag_keys:
        search_results = search_by_tag_key(conn, query_embedding, tag_key, k)
        formatted_results = []
        for result in search_results:
            formatted_result = {
                "book_name": result.get("book_name", ""),
                "chapter_index": result.get("chapter_index", ""),
                "section_index": result.get("section_index", ""),
                "page_index": result.get("page_index"),
                "tag_key": tag_key,
                "similarity_score": result.get("similarity", 0.0),
            }
            formatted_results.append(formatted_result)
        results[tag_key] = formatted_results
    
    conn.close()
    return results

def _format_retrieved_context(results: Dict[str, List[Dict[str, Any]]]) -> str:
    """Format retrieved context from multiple tag keys into a readable string."""
    if not results:
        return "No relevant context found."
    
    formatted_parts = []
    for tag_key, entries in results.items():
        if entries:
            formatted_parts.append(f"\n=== {tag_key.upper()} ===")
            for i, entry in enumerate(entries, 1):
                book = entry.get('book_name', 'Unknown')
                chapter = entry.get('chapter_index', 'N/A')
                section = entry.get('section_index', 'N/A')
                score = entry.get('similarity_score', 0)
                formatted_parts.append(
                    f"{i}. {book}, Chapter {chapter}, Section {section} (similarity: {score:.3f})"
                )
    
    return "\n".join(formatted_parts)

def _generate_diagnosis(patient_case: str, retrieved_context: str, api_key: str = None) -> str:
    """Generate diagnosis using LLM based on patient case and retrieved context."""
    client = _get_client(api_key)
    
    system_prompt = """You are an expert Traditional Chinese Medicine (TCM) practitioner. 
    Based on the patient case and relevant TCM knowledge from classical texts, provide a comprehensive diagnosis.
    Include pattern identification (辨證), treatment principles (治則), and recommended formulas if applicable.
    Respond in both Chinese and English."""
    
    user_prompt = f"""Patient Case:
{patient_case}

Relevant TCM Knowledge:
{retrieved_context}

Please provide a TCM diagnosis based on the above information."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content

def diagnose(patient_case: str, tag_keys: List[str], k: int = DEFAULT_TOP_K, api_key: str = None) -> Dict[str, Any]:
    """Retrieve relevant context and generate TCM diagnosis for a patient case.
    
    This is the main public function that combines retrieval and generation.
    
    Args:
        patient_case: Patient case description in natural language.
        tag_keys: List of tag keys to search (e.g., ['symptoms', 'syndromes', 'formulas']).
        k: Number of results to retrieve per tag key.
        api_key: OpenAI API key.
    
    Returns:
        Dict containing diagnosis, retrieved_context, and metadata.
    """
    retrieved_results = multi_key_query(patient_case, tag_keys, k, api_key)
    
    formatted_context = _format_retrieved_context(retrieved_results)
    
    diagnosis = _generate_diagnosis(patient_case, formatted_context, api_key)
    
    return {
        "diagnosis": diagnosis,
        "retrieved_context": retrieved_results,
        "formatted_context": formatted_context,
        "metadata": {
            "tag_keys_searched": tag_keys,
            "results_per_key": k,
            "total_results": sum(len(v) for v in retrieved_results.values())
        }
    }

def main():
    """Example usage of the query engine."""
    logging.basicConfig(level=logging.INFO)

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is required")
        return

    print("=== Formula Query ===")
    results = query("大陷胸湯", "formulas", k=3, api_key=api_key)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['book_name']} Chapter {result['chapter_index']} - Section {result['section_index']}")
        print()

    print("=== Multi-key Query ===")
    multi_results = multi_key_query("大陷胸湯 nourish yin", ["formulas", "treatments"], k=2, api_key=api_key)
    for tag_key, key_results in multi_results.items():
        print(f"\n{tag_key.upper()} Results:")
        for result in key_results:
            print(f"  - {result['book_name']} Chapter {result['chapter_index']} Section {result['section_index']}")

if __name__ == "__main__":
    main()
