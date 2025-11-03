"""Symptom-only ingestion via GPT-4o per section, preferring parsed JSON if available."""

import logging
import sys

from embeddings import get_batch_embeddings, extract_semantic_symptoms_and_freq
from database import reset_database, insert_section_terms, setup_database

# TEMP: Add project root to path to enable importing from tagging module
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from tagging.parse_books import parse_book


def ingest_all_sections(textbook_path: str = "books/《人紀傷寒論》.txt") -> int:
    """

    """
    parsed_book = parse_book(textbook_path)
    conn = reset_database()
    inserted = 0

    # Process book by chapter and section
    for chapter in parsed_book:
        chapter_idx = chapter["chapter_idx"]
        for section in chapter["sections"]:
            section_idx = section["section_idx"]
            
            # 1. Extract the symptoms and their frequency from the section text
            symptom_freqs = extract_semantic_symptoms_and_freq(section["section_text"])
            # Skip sections that contain no symptoms
            if len(symptom_freqs) == 0:
                continue
            
            # 2. Embed the symptoms
            symptoms_embedded = get_batch_embeddings(list(symptom_freqs.keys()))

            # 3. Add to database
            section_data = {
                "book_name": "《人紀傷寒論》",
                "chapter_index": chapter_idx,
                "section_index": section_idx,
                "page_index": 0,
                "length": sum(symptom_freqs.values()),
            }
            term_rows = []
            for symptom, frequency in symptom_freqs.items():
                term_rows.append({"term": symptom, "freq": frequency, "embedding": symptoms_embedded[symptom]})
            
            conn.execute("BEGIN")
            inserted += insert_section_terms(conn, section_data, term_rows)
            conn.commit()
    
    conn.close()
    return inserted

def main():
    logging.basicConfig(level=logging.INFO)
    setup_database()
    count = ingest_all_sections()
    print(f"Ingested sections: {count}")


if __name__ == "__main__":
    main()
