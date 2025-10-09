"""Data ingestion pipeline for Tag RAG system (SQLite + sqlite-vec)."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

from database import setup_database, insert_section
from embeddings import process_section_tags
from embeddings import TAGS_JSON_PATH

logger = logging.getLogger(__name__)

def load_tags_data(tags_json_path: str = TAGS_JSON_PATH) -> List[Dict[str, Any]]:
    """Load tags data from JSON file."""
    tags_path = Path(tags_json_path)
    if not tags_path.exists():
        raise FileNotFoundError(f"Tags file not found: {tags_json_path}")
    
    with open(tags_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('sections', [])

def process_section(section: Dict[str, Any], api_key: str = None) -> Dict[str, Any]:
    """Process a single section and return data ready for database insertion."""
    section_data = {
        "book_name": section.get("book_name", ""),
        "chapter_idx": section.get("chapter_idx", ""),
        "chapter_title": section.get("chapter_title", ""),
        "section_idx": section.get("section_idx", ""),
        "section_title": section.get("section_title", "")
    }
    
    tag_processing_result = process_section_tags(section, api_key)
    
    return {
        "section_data": section_data,
        "embeddings": tag_processing_result["embeddings"]
    }

def ingest_all_sections(api_key: str = None, tags_json_path: str = TAGS_JSON_PATH) -> int:
    """Ingest all sections from the tags JSON file."""
    logger.info("Starting data ingestion...")
    
    sections = load_tags_data(tags_json_path)
    logger.info(f"Loaded {len(sections)} sections")
    
    conn = setup_database()
    
    inserted_count = 0
    conn.execute("BEGIN")
    for section in tqdm(sections, desc="Processing sections"):
        processed = process_section(section, api_key)

        section_id = insert_section(
            conn,
            processed["section_data"],
            processed["embeddings"]
        )

        if section_id:
            inserted_count += 1
    conn.commit()
    conn.close()
    logger.info(f"Ingestion complete. Inserted: {inserted_count}")
    return inserted_count

def main():
    """Main ingestion function."""
    logging.basicConfig(level=logging.INFO)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is required")
        return
    
    count = ingest_all_sections(api_key)
    print(f"Successfully ingested {count} sections")

if __name__ == "__main__":
    main()
