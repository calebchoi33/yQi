"""Semantic BM25 scorer over per-term sqlite-vec index."""

import os
from typing import List, Dict, Any, Tuple

import numpy as np

from embeddings import (
    get_batch_embeddings,
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
    # Mongo Atlas-style normalization: score = (1 + cosine) / 2
    # sqlite-vec returns cosine distance d, where cosine = 1 - d
    # Therefore score = (1 + (1 - d)) / 2 = 1 - d/2, mapped to [0, 1]
    s = 1.0 - (float(d) / 2.0)
    return s


def _soft_idf(num_sections: int, sims: List[float], df_threshold: float = 0.6) -> float:
    # Only count similarities above threshold to avoid inflating DF with mid-sim sections
    n_t_sem = float(sum(s for s in sims if s > df_threshold))
    idf = float(np.log((num_sections - n_t_sem + 0.5) / (n_t_sem + 0.5)))
    return idf


def _tf_sem(freq_sim_max: float, c_len: int, avg_len: float, k1: float, b: float) -> float:
    denom = freq_sim_max + k1 * (1 - b + b * (c_len / max(1.0, avg_len)))
    return (freq_sim_max / denom) if denom > 0 else 0.0


def _per_section_best(
    conn,
    sections: List[Dict[str, Any]],
    tblob: memoryview,
) -> List[Tuple[Dict[str, Any], float, str, int]]:
    out: List[Tuple[Dict[str, Any], float, str, int]] = []
    for s in sections:
        bm = best_match_for_section(conn, s, tblob)
        sim = _cos_from_distance(bm["distance"])
        best_term = bm.get("term", "")
        freq = count_term_in_section(conn, s, best_term) if best_term else 0
        out.append((s, sim, best_term, freq))
    return out

def score(
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
    api_key: str = None,
    k1: float = 1.2,
    b: float = 0.75,
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
    query_terms: List[str] = extract_query_symptoms(query_text)
    if not query_terms:
        conn.close()
        return []
    # Batch embed extracted symptoms
    term_vecs = get_batch_embeddings(query_terms)
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

        per_section = _per_section_best(conn, sections, tblob)
        sims = [sim for (_, sim, _, _) in per_section]
        idf_sem = _soft_idf(N, sims)

        for (s, sim, best_term, freq) in per_section:
            c_len = max(1, int(s.get("length", 0)))
            freq_sim_max = sim * float(freq) if freq > 0 else 0.0
            tf_sem = _tf_sem(freq_sim_max, c_len, avg_len, k1, b)
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

