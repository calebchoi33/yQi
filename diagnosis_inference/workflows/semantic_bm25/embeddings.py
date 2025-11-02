"""Embedding utilities and per-term tag extraction for Semantic BM25 workflow."""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client_cache = None

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
DEFAULT_TOP_K = 5


def _get_client(api_key: str = None) -> OpenAI:
    global _client_cache
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is required")
    if _client_cache is None:
        _client_cache = OpenAI(api_key=api_key)
    return _client_cache


def get_embedding(text: str, api_key: str = None) -> Optional[np.ndarray]:
    if not text or not text.strip():
        return None
    client = _get_client(api_key)
    r = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    v = np.array(r.data[0].embedding, dtype=np.float32)
    v = v / np.linalg.norm(v)
    return v


def get_embeddings_batch_texts(texts: List[str], api_key: str = None) -> Dict[str, np.ndarray]:
    dedup = [t for t in dict.fromkeys([t for t in texts if t and t.strip()])]
    if not dedup:
        return {}
    client = _get_client(api_key)
    r = client.embeddings.create(model=EMBEDDING_MODEL, input=dedup)
    out: Dict[str, np.ndarray] = {}
    for t, d in zip(dedup, r.data):
        v = np.array(d.embedding, dtype=np.float32)
        v = v / np.linalg.norm(v)
        out[t] = v
    return out


def extract_symptoms_with_frequency(text: str, chapter_index: int, section_index: int, api_key: str = None) -> List[Dict[str, Any]]:
    client = _get_client(api_key)
    system = (
        "Return strict JSON only. Extract symptom phrases that appear in the section and report how many times each appears in this section (count mentions and close paraphrases)."
    )
    prompt = (
        "Task: From the TEXT (this section), list symptoms that appear and how many times each is used in this section.\n"
        "Rules:\n"
        "- Frequency = number of mentions in THIS SECTION, counting obvious paraphrases/near-synonyms and punctuation/whitespace variants.\n"
        "- Use the most concise surface form present in the TEXT and sum counts across its variants.\n"
        "Output JSON exactly in this schema (no extra fields):\n"
        "  {\n"
        "    \"section_id\": \"ch%02d_sec%03d\",\n"
        "    \"tags\": [ {\"tag\": str, \"frequency\": int}, ... ]\n"
        "  }\n"
        f"Section indices: chapter_index={chapter_index}, section_index={section_index}.\n"
        f"TEXT:\n{text}"
    )
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    content = r.choices[0].message.content
    data = json.loads(content)
    items = data.get("tags", [])
    out: List[Dict[str, Any]] = []
    for it in items:
        term = str(it.get("tag", "")).strip()
        freq = int(it.get("frequency", 1))
        if term:
            out.append({"term": term, "freq": freq})
    return out


def extract_query_symptoms(query_text: str, api_key: str = None, max_items: int = 15) -> List[str]:
    client = _get_client(api_key)
    system = "Return strict JSON only. Identify the symptom phrases explicitly present or strongly implied in the USER QUERY."
    prompt = (
        "Output JSON exactly in this schema (no extra fields):\n"
        "  {\n"
        "    \"symptoms\": [str, ...]\n"
        "  }\n"
        f"Rules:\n- Use concise surface forms present in the query.\n- Maximum {max_items} items.\n- Do not invent content.\n\n"
        f"USER QUERY:\n{query_text}"
    )
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    content = r.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        return []
    arr = data.get("symptoms", [])
    out: List[str] = []
    for s in arr:
        t = str(s).strip()
        if t:
            out.append(t)
    if len(out) > max_items:
        out = out[:max_items]
    return out
