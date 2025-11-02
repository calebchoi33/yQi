"""Symptom-only ingestion via GPT-4o per section, preferring parsed JSON if available."""

import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

from embeddings import get_embeddings_batch_texts, extract_symptoms_with_frequency
from database import reset_database, insert_section_terms


def _read_textbook(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _split_chapters_sections(text: str) -> List[Dict[str, Any]]:
    parts = text.split("#CHAPTER")
    out: List[Dict[str, Any]] = []
    ch_idx = 0
    for part in parts:
        blk = part.strip()
        if not blk:
            continue
        ch_idx += 1
        secs = blk.split("#SECTION")
        s_idx = 0
        for sec in secs:
            s = sec.strip()
            if not s:
                continue
            s_idx += 1
            out.append({"chapter_index": ch_idx, "section_index": s_idx, "text": s})
    return out


def ingest_all_sections(api_key: str = None, textbook_path: str = "books/《人紀傷寒論》.txt") -> int:
    book_path = Path(textbook_path)
    parsed_path = book_path.with_name(f"{book_path.stem}_parsed.json")

    spans: List[Dict[str, Any]] = []
    if parsed_path.exists():
        data = json.loads(parsed_path.read_text(encoding="utf-8"))
        for ch in data:
            ch_idx_1 = int(ch.get("chapter_idx", 0)) + 1
            for sec in ch.get("sections", []):
                sec_idx_1 = int(sec.get("section_idx", 0)) + 1
                text = str(sec.get("section_text", "")).strip()
                if not text:
                    continue
                spans.append({"chapter_index": ch_idx_1, "section_index": sec_idx_1, "text": text})
    else:
        txt = _read_textbook(textbook_path)
        spans = _split_chapters_sections(txt)

    conn = reset_database()
    inserted = 0
    for span in tqdm(spans, desc="SemanticBM25 sections"):
        ch = int(span["chapter_index"])
        sec = int(span["section_index"])
        text = span["text"]
        items = extract_symptoms_with_frequency(text, ch, sec, api_key)
        section_data = {
            "book_name": "《人紀傷寒論》",
            "chapter_index": ch,
            "section_index": sec,
            "page_index": 0,
            "length": len(items),
        }
        terms = [str(it.get("term", "")).strip() for it in items if str(it.get("term", "")).strip()]
        vecs = get_embeddings_batch_texts(terms, api_key)
        term_rows: List[Dict[str, Any]] = []
        for it in items:
            term = str(it.get("term", "")).strip()
            if not term:
                continue
            v = vecs.get(term)
            if v is None:
                continue
            freq = int(it.get("freq", 1))
            term_rows.append({"term": term, "freq": freq, "embedding": v})
        if term_rows:
            conn.execute("BEGIN")
            inserted += insert_section_terms(conn, section_data, term_rows) > 0
            conn.commit()
    conn.close()
    return inserted


def main():
    logging.basicConfig(level=logging.INFO)
    api_key = os.getenv("OPENAI_API_KEY")
    count = ingest_all_sections(api_key)
    print(f"Ingested sections: {count}")


if __name__ == "__main__":
    main()
