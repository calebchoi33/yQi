#!/usr/bin/env python3
"""
Retrieval metrics and gold hit@k evaluation for Semantic BM25 (all tags, no tokenization).

Usage:
  python diagnosis_inference/tests/retrieval_metrics_semantic.py \
    --patient-cases "diagnosis_inference/Patient cases - exact matching.docx" \
    --k 3 5 10 --details

Notes:
  - Uses semantic_bm25/query_engine.score() which requires the DB created by
    diagnosis_inference/workflows/semantic_bm25/ingestion.py
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv  # type: ignore
from docx import Document  # type: ignore

_SB_PATH = Path(__file__).resolve().parents[1] / "workflows" / "semantic_bm25"
if str(_SB_PATH) not in sys.path:
    sys.path.append(str(_SB_PATH))
from query_engine import score  # type: ignore
from database import setup_database, get_all_sections  # type: ignore

load_dotenv()


def _parse_patient_cases_txt(txt: str) -> List[Dict]:
    norm = txt.replace("（", "(").replace(")", ")")
    lines = [l.strip() for l in norm.splitlines() if l.strip()]
    cases: List[Dict] = []
    header_re = re.compile(r"^#\s*(\d+)\s*.*\(\s*Ch\s*(\d+)\s*Sec\s*(\d+)\s*\)", re.IGNORECASE)
    i = 0
    while i < len(lines):
        m = header_re.match(lines[i])
        if m:
            idx = int(m.group(1))
            ch = int(m.group(2))
            sec = int(m.group(3))
            j = i + 1
            block: List[str] = []
            while j < len(lines) and not header_re.match(lines[j]):
                block.append(lines[j])
                j += 1
            query_text = " ".join(block).strip()
            cases.append({
                "index": idx,
                "chapter_index": ch,
                "section_index": sec,
                "content": query_text,
            })
            i = j
        else:
            i += 1
    return cases


def load_patient_cases(path: str) -> List[Dict]:
    p = Path(path)
    if p.suffix.lower() == ".txt":
        txt = p.read_text(encoding="utf-8")
        return _parse_patient_cases_txt(txt)
    doc = Document(str(p))
    txt = "\n".join([para.text for para in doc.paragraphs])
    return _parse_patient_cases_txt(txt)


def evaluate_patient_cases(cases: List[Dict], ks: List[int], verbose: bool = False) -> Dict[int, int]:
    if not cases:
        return {k: 0 for k in ks}
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    total = len(cases)
    for i, c in enumerate(cases, 1):
        content = c.get("content", "")
        gold_ch = int(c.get("chapter_index", -1))
        gold_sec_display = int(c.get("section_index", -1))
        gold_comp = (gold_ch, gold_sec_display)
        if verbose:
            preview = content if len(content) <= 100 else content[:100] + "..."
            print(f"[Eval] Case {i}/{total} (Gold Ch {gold_ch} Sec {gold_sec_display}) query len={len(content)}")
            print(f"[Eval] Query preview: {preview}")
        rows = score(content, top_k=max_k, api_key=os.getenv("OPENAI_API_KEY"), verbose=verbose)
        if verbose and rows:
            r0 = rows[0]
            print(f"[Eval] Top-1 => Ch {r0.get('chapter_index')} Sec {r0.get('section_index')} score={r0.get('score'):.4f} term='{r0.get('best_term','')}' sim={r0.get('sim',0.0):.3f}")
        ranking_ids = [
            (int(r.get("chapter_index", -1)), int(r.get("section_index", -1))) for r in rows
        ]
        for k in ks:
            if gold_comp in ranking_ids[:k]:
                hits[k] += 1
    return hits


def dump_patient_case_results(cases: List[Dict], ks: List[int]) -> None:
    if not cases:
        print("No cases to display.")
        return
    max_k = max(ks)
    for c in cases:
        idx = int(c.get("index", -1))
        gold = (int(c.get("chapter_index", -1)), int(c.get("section_index", -1)))
        content = c.get("content", "").strip()
        preview = content if len(content) <= 120 else content[:120] + "..."
        print(f"\nCase #{idx} (Gold: Chapter {gold[0]} Sec {gold[1]}): {preview}")
        rows = score(content, top_k=max_k, api_key=os.getenv("OPENAI_API_KEY"))
        for i, r in enumerate(rows[:5], 1):
            book = r.get("book_name", "")
            ch = r.get("chapter_index", "")
            sec = r.get("section_index", "")
            sc = r.get("score", 0.0)
            sim = r.get("sim", 0.0)
            term = r.get("best_term", "")
            print(f"  {i}. [Book: {book}, Chapter: {ch}, Section: {sec}] (score: {sc:.4f}, sim: {sim:.3f}, term: {term})")
        if not rows:
            print("  (No retrieval results. Ensure the semantic_bm25 DB is ingested.)")


def index_alignment_report(cases: List[Dict]) -> None:
    conn = setup_database()
    sections = get_all_sections(conn)
    conn.close()
    db_pairs = set((int(s.get("chapter_index", -1)), int(s.get("section_index", -1))) for s in sections)
    missing = []
    for c in cases:
        pair = (int(c.get("chapter_index", -1)), int(c.get("section_index", -1)))
        if pair not in db_pairs:
            missing.append({"index": int(c.get("index", -1)), "pair": pair})
    total = len(cases)
    print("INDEX ALIGNMENT CHECK:")
    print(f"Total cases: {total}")
    print(f"Present in DB: {total - len(missing)}")
    print(f"Missing in DB: {len(missing)}")
    if missing:
        print("Examples of missing (first 10):")
        for m in missing[:10]:
            print(f"  Case #{m['index']}: (Ch {m['pair'][0]} Sec {m['pair'][1]}) not found in DB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient-cases", required=True, help="Path to Patient Cases .docx or .txt")
    ap.add_argument("--k", nargs="*", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--details", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="Print progress and debug info", default=False)
    ap.add_argument("--check-index", action="store_true", help="Only check index alignment and exit", default=False)
    args = ap.parse_args()

    ks = args.k
    cases = load_patient_cases(args.patient_cases)

    if args.check_index:
        index_alignment_report(cases)
        return
    hits = evaluate_patient_cases(cases, ks, verbose=args.verbose)

    print("RESULTS (Semantic BM25, Patient Cases gold):")
    print(f"Total # cases = {len(cases)}")
    for k in ks:
        print(f"k={k}: {hits[k]}/{len(cases)}")
    if args.details:
        dump_patient_case_results(cases, ks)


if __name__ == "__main__":
    main()
