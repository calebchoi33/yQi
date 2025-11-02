"""Semantic BM25 scorer over per-term sqlite-vec index."""

import os
from typing import List, Dict, Any, Tuple

import numpy as np

from embeddings import (
    get_embeddings_batch_texts,
    extract_query_symptoms,
    DEFAULT_TOP_K,
)
from database import (
    setup_database,
    get_all_sections,
    best_match_for_section,
    avg_section_length,
    count_term_in_section,
)


def _to_f32_blob(vector: np.ndarray) -> memoryview:
    if vector.dtype != np.float32:
        vector = vector.astype(np.float32)
    return memoryview(vector.tobytes())


def _cos_from_distance(d: float) -> float:
    if d is None:
        return 0.0
    s = 1.0 - float(d)
    if s < 0.0:
        return 0.0
    if s > 1.0:
        return 1.0
    return s


def score(
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
    api_key: str = None,
    k1: float = 1.2,
    b: float = 0.75,
    tau: float = 0.3,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    conn = setup_database()

    sections = get_all_sections(conn)
    N = len(sections)
    if N == 0:
        conn.close()
        return []
    avg_len = avg_section_length(conn) or 1.0

    # LLM-based extraction of symptoms directly from the query
    query_terms: List[str] = extract_query_symptoms(query_text, api_key=api_key)
    if not query_terms:
        conn.close()
        return []
    # Batch embed extracted symptoms
    term_vecs = get_embeddings_batch_texts(query_terms, api_key)
    if not term_vecs:
        conn.close()
        return []

    if verbose or os.getenv("SEMBM25_VERBOSE") == "1":
        print("[SemanticBM25] Sections:", N, "AvgLen:", f"{avg_len:.2f}")
        print("[SemanticBM25] Query symptoms (", len(query_terms), "):", query_terms[:20])

    scores: Dict[Tuple[str, str, str, int], float] = {}
    meta: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    diag_best: Dict[Tuple[str, str, str, int], Tuple[float, str]] = {}

    for t, tv in term_vecs.items():
        tblob = _to_f32_blob(tv)

        per_section: List[Tuple[Dict[str, Any], float, str, int]] = []
        for s in sections:
            bm = best_match_for_section(conn, s, tblob)
            sim = _cos_from_distance(bm["distance"])
            best_term = bm.get("term", "")
            freq = count_term_in_section(conn, s, best_term) if best_term else 0
            per_section.append((s, sim, best_term, freq))

        n_t_sem = float(sum(sim for (_, sim, _, _) in per_section))
        idf_sem = float(np.log((N - n_t_sem + 0.5) / (n_t_sem + 0.5)))
        if idf_sem < 0.0:
            idf_sem = 0.0

        for (s, sim, best_term, freq) in per_section:
            c_len = max(1, int(s.get("length", 0)))
            freq_sim_max = sim * float(freq) if freq > 0 else 0.0
            denom = freq_sim_max + k1 * (1 - b + b * (c_len / avg_len))
            tf_sem = (freq_sim_max / denom) if denom > 0 else 0.0
            sc_inc = idf_sem * tf_sem

            key = (
                str(s.get("book_name", "")),
                str(s.get("chapter_index", "")),
                str(s.get("section_index", "")),
                int(s.get("page_index", 0)),
            )
            scores[key] = scores.get(key, 0.0) + float(sc_inc)
            if key not in meta:
                meta[key] = {
                    "book_name": key[0],
                    "chapter_index": key[1],
                    "section_index": key[2],
                    "page_index": key[3],
                }
            prev = diag_best.get(key)
            if prev is None or sim > prev[0]:
                diag_best[key] = (sim, best_term)

    results: List[Dict[str, Any]] = []
    for key, sc in scores.items():
        s_meta = meta[key]
        best = diag_best.get(key, (0.0, ""))
        results.append({
            "book_name": s_meta["book_name"],
            "chapter_index": s_meta["chapter_index"],
            "section_index": s_meta["section_index"],
            "page_index": s_meta["page_index"],
            "score": float(sc),
            "sim": float(best[0]),
            "best_term": best[1],
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    conn.close()
    return results[:top_k]

