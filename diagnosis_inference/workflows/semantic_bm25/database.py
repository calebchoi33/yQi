"""SQLite database for Semantic BM25 workflow using a single vec0 table."""

import sqlite3
from typing import List, Dict, Any, Tuple

import numpy as np

import sqlite_vec

EMBEDDING_DIMENSION = 1536


def _to_f32_blob(vector: np.ndarray) -> memoryview:
    if vector.dtype != np.float32:
        vector = vector.astype(np.float32)
    return memoryview(vector.tobytes())


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    if hasattr(conn, "enable_load_extension"):
        conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    cur = conn.cursor()
    cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp._vec_verify USING vec0(x float[4]);")
    cur.execute("DROP TABLE IF EXISTS temp._vec_verify;")


def connect(db_path: str = "semantic_bm25_vec.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    _load_sqlite_vec(conn)
    return conn


def setup_database(db_path: str = "semantic_bm25_vec.db") -> sqlite3.Connection:
    conn = connect(db_path)
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS terms_joined USING vec0(
            book_name TEXT,
            chapter_index TEXT,
            section_index TEXT,
            page_index INTEGER,
            length INTEGER,
            term TEXT,
            freq INTEGER,
            v float[{EMBEDDING_DIMENSION}] distance_metric=cosine
        );
        """
    )
    conn.commit()
    return conn


def reset_database(db_path: str = "semantic_bm25_vec.db") -> sqlite3.Connection:
    conn = connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS terms_joined")
    cur.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS terms_joined USING vec0(
            book_name TEXT,
            chapter_index TEXT,
            section_index TEXT,
            page_index INTEGER,
            length INTEGER,
            term TEXT,
            freq INTEGER,
            v float[{EMBEDDING_DIMENSION}] distance_metric=cosine
        );
        """
    )
    conn.commit()
    return conn


def insert_section_terms(conn: sqlite3.Connection, section_data: Dict[str, Any], term_rows: List[Dict[str, Any]]) -> int:
    cur = conn.cursor()
    count = 0
    for row in term_rows:
        vec = row["embedding"]
        cur.execute(
            """
            INSERT INTO terms_joined (book_name, chapter_index, section_index, page_index, length, term, freq, v)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(section_data.get("book_name", "")),
                str(section_data.get("chapter_index", section_data.get("chapter_idx", ""))),
                str(section_data.get("section_index", section_data.get("section_idx", ""))),
                int(section_data.get("page_index", 0)),
                int(section_data.get("length", 0)),
                str(row.get("term", "")),
                int(row.get("freq", 1)),
                _to_f32_blob(vec),
            ],
        )
        count += 1
    return count


def get_all_sections(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT book_name, chapter_index, section_index, page_index, MAX(length) as length
        FROM terms_joined
        GROUP BY book_name, chapter_index, section_index, page_index
        """
    )
    rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({
            "book_name": r[0],
            "chapter_index": r[1],
            "section_index": r[2],
            "page_index": r[3],
            "length": r[4],
        })
    return out


def avg_section_length(conn: sqlite3.Connection) -> float:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT AVG(x.len) FROM (
            SELECT MAX(length) AS len
            FROM terms_joined
            GROUP BY book_name, chapter_index, section_index, page_index
        ) x
        """
    )
    v = cur.fetchone()[0]
    return float(v) if v is not None else 0.0


def best_match_for_section(conn: sqlite3.Connection, section: Dict[str, Any], qblob: memoryview) -> Dict[str, Any]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT distance, term
        FROM terms_joined
        WHERE book_name = ? AND chapter_index = ? AND section_index = ? AND page_index = ?
          AND v MATCH ?
          AND k = 1
        ORDER BY distance ASC
        """,
        (
            str(section.get("book_name", "")),
            str(section.get("chapter_index", "")),
            str(section.get("section_index", "")),
            int(section.get("page_index", 0)),
            qblob,
        ),
    )
    row = cur.fetchone()
    if not row:
        return {"distance": None, "term": ""}
    return {"distance": float(row[0]), "term": row[1]}


def count_term_in_section(conn: sqlite3.Connection, section: Dict[str, Any], term: str) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(SUM(freq), 0) FROM terms_joined
        WHERE book_name = ? AND chapter_index = ? AND section_index = ? AND page_index = ? AND term = ?
        """,
        (
            str(section.get("book_name", "")),
            str(section.get("chapter_index", "")),
            str(section.get("section_index", "")),
            int(section.get("page_index", 0)),
            term,
        ),
    )
    return int(cur.fetchone()[0])


def get_all_distinct_terms(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT term FROM terms_joined
        WHERE term IS NOT NULL AND term <> ''
        """
    )
    rows = cur.fetchall()
    return [r[0] for r in rows]


def best_matches_for_query_vector(conn: sqlite3.Connection, qblob: memoryview, k: int) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT distance, book_name, chapter_index, section_index, page_index, term
        FROM terms_joined
        WHERE v MATCH ? AND k = ?
        ORDER BY distance ASC
        """,
        (qblob, int(k)),
    )
    rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({
            "distance": float(r[0]),
            "book_name": r[1],
            "chapter_index": r[2],
            "section_index": r[3],
            "page_index": int(r[4]),
            "term": r[5],
        })
    return out
