#!/usr/bin/env python3
"""Pytest suite for Tag RAG Proper (SQLite + sqlite-vec).

These tests cover:
- Database setup and schema
- Ingestion pipeline
- Query functions (single and multi-key)
- End-to-end diagnose()

Note: Ingestion calls the OpenAI Embeddings API and can take time/cost credits.
Set the environment variable OPENAI_API_KEY before running.
"""

import os
import sys
from pathlib import Path
import pytest

# Ensure workflows/tag_rag_proper is importable
ROOT = Path(__file__).resolve().parents[1]
TAG_RAG_PROPER = ROOT / "workflows" / "tag_rag_proper"
if str(TAG_RAG_PROPER) not in sys.path:
    sys.path.insert(0, str(TAG_RAG_PROPER))

from database import setup_database, get_section_columns
from ingestion import ingest_all_sections
from query_engine import query, multi_key_query, diagnose
from embeddings import TAG_KEYS


@pytest.fixture(scope="session")
def api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set; skipping Tag RAG Proper tests that require API access.")
    return key


def test_database_structure_smoke():
    """Database setup should produce the vec_joined table with metadata cols."""
    conn = setup_database()
    cols = get_section_columns(conn)
    conn.close()

    assert isinstance(cols, list) and len(cols) >= 4
    # Minimal metadata columns expected
    for col in ("book_name", "chapter_index", "section_index", "page_index"):
        assert col in cols


def test_ingestion_pipeline(api_key):
    """Ingest sections into vec_joined table. Should insert >= 1 section."""
    count = ingest_all_sections(api_key)
    assert isinstance(count, int)
    assert count >= 1


def test_single_key_query(api_key):
    """Basic retrieval by a single key should return a list of dict results."""
    results = query("大陷胸湯", "formulas", k=3, api_key=api_key)
    assert isinstance(results, list)
    # Results may be empty depending on data/ingestion, but shape should be dicts
    for item in results:
        assert isinstance(item, dict)
        assert "book_name" in item and "similarity_score" in item


def test_multi_key_query(api_key):
    """Search across multiple keys and validate grouped results structure."""
    keys = ["formulas", "symptoms", "organs"]
    grouped = multi_key_query("大陷胸湯 with heart symptoms", keys, k=2, api_key=api_key)
    assert isinstance(grouped, dict)
    for k in keys:
        assert k in grouped
        assert isinstance(grouped[k], list)


def test_end_to_end_diagnose(api_key):
    """End-to-end: retrieve context and generate diagnosis using gpt-4o."""
    patient_case = "患者發熱惡寒，頭痛身痛，脈浮，口渴不甚。"
    keys = ["symptoms", "syndromes", "formulas"]
    result = diagnose(patient_case=patient_case, tag_keys=keys, k=2, api_key=api_key)

    assert isinstance(result, dict)
    assert "diagnosis" in result
    assert "formatted_context" in result
    assert "retrieved_context" in result and isinstance(result["retrieved_context"], dict)
