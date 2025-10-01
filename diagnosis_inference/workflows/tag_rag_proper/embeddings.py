"""Embedding utilities and tag processing for Tag RAG system."""

import os
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_client_cache = None

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

TAGS_JSON_PATH = "《人紀傷寒論》_tags.json"

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

DEFAULT_TOP_K = 5

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
    """Process section data and return embeddings for each tag key."""
    tag_texts = {}
    embeddings = {}
    
    for key, value in section_data.items():
        if isinstance(value, list) and key not in ["chapter_idx", "section_idx"]:
            tag_text = create_tag_text(value)
            tag_texts[key] = tag_text
            
            if tag_text:
                embedding = get_embedding(tag_text, api_key)
                embeddings[key] = embedding
            else:
                embeddings[key] = None
    
    return {
        "tag_texts": tag_texts,
        "embeddings": embeddings
    }
