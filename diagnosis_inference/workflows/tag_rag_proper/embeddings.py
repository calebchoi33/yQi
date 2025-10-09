"""Embedding utilities and tag processing for Tag RAG system."""

import os
import numpy as np
from typing import Dict, List, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client_cache = None

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

TAGS_JSON_PATH = "../../../tagging/output/《人紀傷寒論》_tags.json"

TAG_KEYS = [
    "formulas",
    "syndromes",
    "treatments",
    "pathogens",
    "organs",
    "herbs",
    "symptoms",
    "pulses",
    "acupoints",
    "meridians",
    "elements"
]

DEFAULT_TOP_K = 15

# Map keys coming from the tagging JSON (e.g., "symptom_tags")
# to the canonical schema keys used as vector columns (e.g., "symptoms").
# Only keys present in TAG_KEYS will be embedded and inserted.
KEY_MAP = {
    # core
    "symptom_tags": "symptoms",
    "syndrome_tags": "syndromes",
    "formula_tags": "formulas",
    "treatment_tags": "treatments",
    # others
    "pathogen_tags": "pathogens",
    "organ_tags": "organs",
    "herb_tags": "herbs",
    "pulse_tags": "pulses",
    "acupoint_tags": "acupoints",
    "meridian_tags": "meridians",
    "element_tags": "elements",
}

def _get_client(api_key: str = None) -> OpenAI:
    """Get or create cached OpenAI client."""
    global _client_cache
    api_key = api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key is required")
    if _client_cache is None:
        _client_cache = OpenAI(api_key=api_key)
    return _client_cache

def get_embedding(text: str, api_key: str = None) -> np.ndarray:
    """Get embedding for a single text string."""
    if not text.strip():
        return None
    
    client = _get_client(api_key)
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    
    embedding = np.array(response.data[0].embedding, dtype=np.float32)
    
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding

def create_tag_text(tags: List[Dict[str, str]]) -> str:
    """Create a single text string from a list of tags."""
    if not tags:
        return ""
    
    terms = set()
    for tag in tags:
        zh_term = tag.get("name_zh", "").strip()
        en_term = tag.get("name_en", "").strip()
        
        if zh_term:
            terms.add(zh_term)
        if en_term:
            terms.add(en_term)
    
    return " ".join(sorted(terms))

def process_section_tags(section_data: Dict[str, Any], api_key: str = None) -> Dict[str, Any]:
    """Process section data and return embeddings keyed by schema TAG_KEYS.

    This function normalizes incoming tag family keys from the JSON (e.g.,
    "symptom_tags") into the canonical schema keys defined by TAG_KEYS
    (e.g., "symptoms"). Unknown keys are ignored. For known keys that have
    no terms, we emit None so the DB layer can handle placeholders.
    """
    tag_texts: Dict[str, str] = {}
    embeddings_out: Dict[str, Optional[np.ndarray]] = {}

    # Initialize all expected keys with None to make downstream handling simple
    for k in TAG_KEYS:
        embeddings_out[k] = None

    for raw_key, value in section_data.items():
        if not isinstance(value, list):
            continue
        mapped_key = KEY_MAP.get(raw_key)
        if not mapped_key or mapped_key not in TAG_KEYS:
            continue
        tag_text = create_tag_text(value)
        tag_texts[mapped_key] = tag_text
        if tag_text:
            embeddings_out[mapped_key] = get_embedding(tag_text, api_key)
        else:
            embeddings_out[mapped_key] = None

    return {
        "tag_texts": tag_texts,
        "embeddings": embeddings_out,
    }
