"""Query engine for Tag RAG system (SQLite + sqlite-vec)."""

import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Any

from database import setup_database, search_by_tag_key
from embeddings import get_embedding, DEFAULT_TOP_K, _get_client
from gather_relevant_text import get_relevant_sections

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

 

def _format_enriched_context(enriched: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format enriched results (with chunk_text) into a list of entries.

    Each entry dict contains: book_name, chapter_index, section_index, similarity, tag_key, chunk_text
    """
    items: List[Dict[str, Any]] = []
    for e in enriched:
        items.append({
            "book_name": e.get("book_name", "") or "《人紀傷寒論》",
            "chapter_index": e.get("chapter_index", "N/A"),
            "section_index": e.get("section_index", "N/A"),
            "similarity": float(e.get("similarity", 0.0)),
            "tag_key": e.get("tag_key", ""),
            "chunk_text": e.get("chunk_text", ""),
        })
    return items

def _context_list_to_str(items: List[Dict[str, Any]]) -> str:
    """Compact string rendering of context items for the LLM prompt with citation + snippet."""
    if not items:
        return ""
    lines = []
    for i, it in enumerate(items, 1):
        citation = (
            f"[Book: {it.get('book_name')}, Chapter: {it.get('chapter_index')}, "
            f"Section: {it.get('section_index')}] (similarity: {it.get('similarity', 0):.3f})"
        )
        snippet = (it.get("chunk_text") or "").strip()
        if len(snippet) > 800:
            snippet = snippet[:800] + "..."
        lines.append(f"{i}. {citation}\n{snippet}")
    return "\n".join(lines)

def _generate_diagnosis(patient_case: str, retrieved_context: str, api_key: str = None) -> str:
    """Generate diagnosis using LLM based on patient case and retrieved context."""
    client = _get_client(api_key)
    
    system_prompt = (
        "You are an expert Traditional Chinese Medicine (TCM) practitioner. "
        "Answer PRIMARILY and EXCLUSIVELY using the provided context from classical texts. "
        "If the provided context is insufficient to answer, respond exactly: no context found. If there is no provided context, respond exactly: no context. If the provided context is not relevant, say so, and explain how."
        "Every factual claim MUST include a citation to the provided context using the format [Book: <book_name>, Chapter: <chapter_index>, Section: <section_index>]. "
        "Provide a comprehensive diagnosis including: 1) pattern identification (辨證), 2) treatment principles (治則), and 3) recommended formulas if applicable. "
        "Respond ONLY in Chinese. Keep the answers short and concise."
    )
    
    user_prompt = (
        "Patient Case:\n"
        f"{patient_case}\n\n"
        "Relevant TCM Knowledge (use these passages for all claims; cite each claim):\n"
        f"{retrieved_context}\n\n"
        "Instructions: Use ONLY the above context for factual claims. If the context is insufficient to answer, respond with 'no context found'. If there is no provided context, respond with 'no context'. Respond ONLY in Chinese. Keep the answers short and concise."
    )
    
    response = client.chat.completions.create(
        model="gpt-4o",
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

    effective_keys = ["symptoms"]
    retrieved_results = multi_key_query(patient_case, effective_keys, k, api_key)

    MIN_SIMILARITY = 0.40
    filtered_results: Dict[str, List[Dict[str, Any]]] = {}
    excluded_keys: List[str] = []
    for key, rows in retrieved_results.items():
        if not rows:
            excluded_keys.append(key)
            continue
        best = max(float(r.get("similarity_score", 0.0)) for r in rows)
        if best >= MIN_SIMILARITY:
            filtered_results[key] = rows
        else:
            excluded_keys.append(key)

    enriched_items = _enrich_results_with_text(filtered_results, api_key=api_key, translate=False)
    context_items = _format_enriched_context(enriched_items)
    context_str = _context_list_to_str(context_items)

    diagnosis = _generate_diagnosis(patient_case, context_str, api_key)
    
    return {
        "diagnosis": diagnosis,
        "retrieved_context": filtered_results,
        "formatted_context": context_items,
        "metadata": {
            "tag_keys_searched": effective_keys,
            "results_per_key": k,
            "total_results": sum(len(v) for v in filtered_results.values()),
            "excluded_keys_for_low_similarity": excluded_keys,
            "min_similarity_threshold": MIN_SIMILARITY,
        }
    }

def _resolve_book_path(book_name: str) -> Path:
    """Resolve the source book file path from a result's book_name.

    Currently defaults to the canonical parsed book in the repo's books/ folder.
    Adjust here if multiple books are supported later.
    """
    root = Path(__file__).resolve().parents[3]
    return root / "books" / "《人紀傷寒論》.txt"

def _safe_int(value: Any, default: int = 0) -> int:
    return int(value)

def _translate_to_english(text: str, api_key: str = None) -> str:
    """Translate Chinese (or mixed) text to English using the OpenAI Chat API.

    Keeps it concise and faithful to the source. Returns empty string if input is empty.
    """
    if not text.strip():
        return ""
    client = _get_client(api_key)
    resp = client.chat.completions.create(
        model=os.getenv("TRANSLATE_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "You are a professional translator. Translate the user content to accurate, clear English. Do not add commentary."},
            {"role": "user", "content": text[:8000]},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""

def _enrich_results_with_text(
    results_by_key: Dict[str, List[Dict[str, Any]]],
    api_key: str = None,
    translate: bool = False,
) -> List[Dict[str, Any]]:
    """Flatten grouped results and attach chunk text for each row.

    Returns a list of dicts with: chapter_index, section_index, similarity, chunk_text, tag_key
    """
    enriched: List[Dict[str, Any]] = []
    for tag_key, entries in results_by_key.items():
        for entry in entries:
            chapter_idx = _safe_int(entry.get("chapter_index", 0))
            section_idx = _safe_int(entry.get("section_index", 0))
            sim = float(entry.get("similarity_score", 0.0))
            book_name = entry.get("book_name", "")
            book_path = _resolve_book_path(book_name)
            chunk_text = get_relevant_sections(str(book_path), chapter_idx, section_idx)
            english_text = _translate_to_english(chunk_text, api_key) if (translate and chunk_text) else ""
            enriched.append({
                "tag_key": tag_key,
                "chapter_index": chapter_idx,
                "section_index": section_idx,
                "similarity": sim,
                "chunk_text": chunk_text,
                "english_text": english_text,
            })
    return enriched

def export_queries_to_json(
    queries: List[Dict[str, Any]],
    output_path: str,
    api_key: str = None,
    translate: bool = False,
) -> str:
    """Run multiple queries, enrich with source text, and write JSON to output_path.

    Each query dict supports keys:
      - query_text: str (required)
      - tag_keys: List[str] (required)
      - k: int (optional; defaults to DEFAULT_TOP_K)

    Output JSON structure:
    {
      "queries": [
        {
          "query_text": str,
          "tag_keys": [...],
          "k": int,
          "results": [
            {"chapter_index": int, "section_index": int, "similarity": float, "chunk_text": str, "tag_key": str}
          ]
        }
      ]
    }
    """
    all_output = {"queries": []}
    for q in queries:
        text = q.get("query_text", "").strip()
        tag_keys = q.get("tag_keys", [])
        k = int(q.get("k", DEFAULT_TOP_K))
        if not text or not tag_keys:
            continue
        grouped = multi_key_query(text, tag_keys, k, api_key)
        enriched = _enrich_results_with_text(grouped, api_key, translate=translate)
        all_output["queries"].append({
            "query_text": text,
            "tag_keys": tag_keys,
            "k": k,
            "results": enriched,
        })

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_output, f, ensure_ascii=False, indent=2)
    return str(out_path)

def main():
    """Example usage of the query engine."""
    logging.basicConfig(level=logging.INFO)

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is required")
        return

    sample_queries = [
        {"query_text": "大陷胸湯", "tag_keys": ["formulas"], "k": 3},
        {"query_text": "大陷胸湯 nourish yin", "tag_keys": ["formulas", "treatments"], "k": 2},
    ]
    out_file = Path(__file__).resolve().parent / "query_results.json"
    export_path = export_queries_to_json(sample_queries, str(out_file), api_key, translate=False)
    print(f"Wrote enriched query results to: {export_path}")

if __name__ == "__main__":
    main()
