"""Embedding utilities and tag processing for Tag RAG system."""

import os
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Configuration constants
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small

# Database configuration
DB_NAME = "tag_rag_db"
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = 5432
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
TAGS_JSON_PATH = "《人紀傷寒論》_tags.json"

# Tag families we want to search (complete set from JSON)
TAG_FAMILIES = {
    "formulas": {
        "vector_column": "formulas_vec",
        "keywords": ["湯", "散", "丸", "膏", "方", "劑", "藥", "decoction", "powder", "pill", "paste", "formula", "prescription", "medicine", "herb"]
    },
    "syndromes": {
        "vector_column": "syndromes_vec",
        "keywords": ["證", "症候群", "病", "太陽", "陽明", "少陽", "太陰", "少陰", "厥陰", "syndrome", "pattern", "disease", "taiyang", "yangming", "shaoyang", "taiyin", "shaoyin", "jueyin"]
    },
    "treatments": {
        "vector_column": "treatments_vec",
        "keywords": ["治", "療", "法", "術", "補", "瀉", "溫", "清", "汗", "吐", "下", "和", "treatment", "therapy", "method", "tonify", "sedate", "warm", "clear", "sweat", "vomit", "purge", "harmonize"]
    },
    "pathogens": {
        "vector_column": "pathogens_vec",
        "keywords": ["邪", "風", "寒", "暑", "濕", "燥", "火", "熱", "痰", "瘀", "氣滯", "血瘀", "pathogen", "wind", "cold", "summer", "heat", "dampness", "dryness", "fire", "phlegm", "stasis", "qi", "blood"]
    },
    "organs": {
        "vector_column": "organs_vec", 
        "keywords": ["心", "肝", "脾", "肺", "腎", "胃", "腸", "膽", "膀胱", "三焦", "心包", "小腸", "大腸", "heart", "liver", "spleen", "lung", "kidney", "stomach", "intestine", "gallbladder", "bladder", "triple", "heater", "pericardium", "organ"]
    },
    "herbs": {
        "vector_column": "herbs_vec",
        "keywords": ["草", "根", "莖", "葉", "花", "果", "種", "皮", "藥材", "本草", "herb", "root", "stem", "leaf", "flower", "fruit", "seed", "bark", "medicinal", "material"]
    },
    "symptoms": {
        "vector_column": "symptoms_vec",
        "keywords": ["症狀", "症", "痛", "熱", "寒", "汗", "咳", "喘", "嘔", "瀉", "便", "尿", "頭", "眼", "耳", "鼻", "口", "舌", "咽", "胸", "腹", "背", "腰", "四肢", "皮膚", "fever", "pain", "headache", "nausea", "vomiting", "diarrhea", "constipation", "cough", "shortness", "breath", "chest", "abdominal", "back", "limb", "skin", "symptom"]
    },
    "pulses": {
        "vector_column": "pulses_vec",
        "keywords": ["脈", "浮", "沉", "遲", "數", "滑", "澀", "弦", "緊", "緩", "細", "大", "長", "短", "pulse", "floating", "sinking", "slow", "rapid", "slippery", "rough", "wiry", "tight", "moderate", "thin", "large", "long", "short"]
    },
    "acupoints": {
        "vector_column": "acupoints_vec",
        "keywords": ["穴", "點", "位", "針", "灸", "按", "摩", "acupoint", "point", "location", "needle", "moxibustion", "massage", "pressure"]
    },
    "meridians": {
        "vector_column": "meridians_vec",
        "keywords": ["經", "絡", "脈", "道", "路", "徑", "meridian", "channel", "pathway", "vessel", "route"]
    },
    "elements": {
        "vector_column": "elements_vec",
        "keywords": ["五行", "木", "火", "土", "金", "水", "運", "氣", "element", "wood", "fire", "earth", "metal", "water", "movement", "qi"]
    },
    "tongues": {
        "vector_column": "tongues_vec",
        "keywords": ["舌", "苔", "質", "色", "形", "態", "tongue", "coating", "body", "color", "shape", "texture"]
    }
}

# Search Configuration
DEFAULT_TOP_K = 5
SIMILARITY_THRESHOLD = 0.7

def get_embedding(text: str, api_key: str = None) -> np.ndarray:
    """Get embedding for a single text string."""
    if not text.strip():
        return None
    
    api_key = api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    client = OpenAI(api_key=api_key)
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    
    embedding = np.array(response.data[0].embedding, dtype=np.float32)
    
    # Normalize to unit length for cosine similarity
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding

def get_embeddings_batch(texts: List[str], api_key: str = None) -> List[Optional[np.ndarray]]:
    """Get embeddings for multiple texts in batch."""
    if not texts:
        return []
    
    api_key = api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    client = OpenAI(api_key=api_key)
    
    # Filter out empty texts but keep track of indices
    non_empty_texts = []
    text_indices = []
    
    for i, text in enumerate(texts):
        if text.strip():
            non_empty_texts.append(text)
            text_indices.append(i)
    
    if not non_empty_texts:
        return [None] * len(texts)
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=non_empty_texts
    )
    
    # Create result array with None for empty texts
    results = [None] * len(texts)
    
    for i, embedding_data in enumerate(response.data):
        original_index = text_indices[i]
        embedding = np.array(embedding_data.embedding, dtype=np.float32)
        
        # Normalize to unit length
        embedding = embedding / np.linalg.norm(embedding)
        
        results[original_index] = embedding
    
    return results

def categorize_tags(section_data: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """Categorize tags into families based on the actual data structure."""
    categorized = {}
    
    # Extract all tag families that exist in the section data
    for family in TAG_FAMILIES.keys():
        categorized[family] = section_data.get(family, [])
    
    return categorized

def create_tag_text(tags: List[Dict[str, str]]) -> str:
    """Create a single text string from a list of tags."""
    if not tags:
        return ""
    
    # Combine Chinese and English names, deduplicate
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
    """Process section data and return categorized data and embeddings."""
    # Categorize tags from section data
    categorized_tags = categorize_tags(section_data)
    
    # Create text strings for each family
    tag_texts = {}
    embeddings = {}
    
    for family, family_tags in categorized_tags.items():
        # Create text for embedding
        tag_text = create_tag_text(family_tags)
        tag_texts[family] = tag_text
        
        # Get embedding
        if tag_text:
            embedding = get_embedding(tag_text, api_key)
            embeddings[family] = embedding
        else:
            embeddings[family] = None
    
    return {
        "tag_texts": tag_texts,
        "embeddings": embeddings,
        "categorized_tags": categorized_tags
    }
