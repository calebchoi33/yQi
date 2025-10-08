"""SQLite database with sqlite-vec for Tag RAG system (single joined table).

Schema uses a fixed set of vector columns for readability, sourced from
`embeddings.TAG_KEYS`.

This module performs a preflight check to ensure the `sqlite-vec` package
is available and that the `vec0` virtual table can be created. If loading
or verification fails, a clear RuntimeError is raised with remediation
instructions.
"""

import sqlite3
from typing import List, Dict, Any

import numpy as np

import sqlite_vec

from embeddings import EMBEDDING_DIMENSION, TAG_KEYS

_table_info_cache = None

def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec and verify vec0 is available on this connection."""
    if hasattr(conn, "enable_load_extension"):
        conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    cur = conn.cursor()
    cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp._vec_verify USING vec0(x float[4]);")
    cur.execute("DROP TABLE IF EXISTS temp._vec_verify;")

def connect(db_path: str = "tag_rag_vec.db") -> sqlite3.Connection:
    """Open a SQLite connection and load+verify sqlite-vec (vec0) extension.

    Requires the `sqlite-vec` Python package (bundles the vec0 extension).
    Raises RuntimeError with guidance if loading/verification fails.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")

    _load_sqlite_vec(conn)
    return conn


def setup_database(db_path: str = "tag_rag_vec.db") -> sqlite3.Connection:
    """Create database connection and ensure single-table schema exists.

    This function creates the `vec_joined` virtual table with metadata columns
    and one vector column per tag key defined in `embeddings.TAG_KEYS`.
    """
    conn = connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vec_joined'")
    if cur.fetchone():
        return conn

    tag_keys: List[str] = sorted(TAG_KEYS)

    meta_cols = [
        "book_name TEXT",
        "chapter_index TEXT",
        "section_index TEXT",
        "page_index INTEGER",
    ]
    vec_cols = [f"{k} float[{EMBEDDING_DIMENSION}]" for k in tag_keys]
    flag_cols = [f"{k}_has_data INTEGER" for k in tag_keys]
    column_defs = ", ".join(meta_cols + vec_cols + flag_cols) if vec_cols else ", ".join(meta_cols)

    cur.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_joined USING vec0({column_defs});")
    conn.commit()
    global _table_info_cache
    _table_info_cache = None
    return conn


def _to_f32_blob(vector: np.ndarray) -> memoryview:
    """Convert a numpy float32 vector to a BLOB acceptable by sqlite-vec."""
    if vector.dtype != np.float32:
        vector = vector.astype(np.float32)
    return memoryview(vector.tobytes())


def _get_table_columns(conn: sqlite3.Connection) -> List[str]:
    """Get cached table columns or fetch if not cached."""
    global _table_info_cache
    if _table_info_cache is None:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(vec_joined)")
        _table_info_cache = [row[1] for row in cur.fetchall()]
    return _table_info_cache

def insert_section(conn: sqlite3.Connection, section_data: Dict[str, Any], tag_vectors: Dict[str, np.ndarray]) -> int:
    """Insert a section row into the single joined vec table with metadata + vectors.

    Args:
        conn: Database connection.
        section_data: Metadata dict with book_name, chapter_idx/ chapter_index, section_idx/ section_index, page_index.
        tag_vectors: Dict mapping tag keys (e.g., 'formulas') to numpy embedding arrays. If a tag key is missing or None, it will be stored as NULL.

    Returns:
        The rowid of the inserted record (serves as the section id).
    """
    cur = conn.cursor()

    cols = _get_table_columns(conn)

    meta_book = str(section_data.get("book_name", ""))
    meta_chap = str(section_data.get("chapter_index", section_data.get("chapter_idx", "")))
    meta_sect = str(section_data.get("section_index", section_data.get("section_idx", "")))
    meta_page = section_data.get("page_index", 0)

    col_names: List[str] = ["book_name", "chapter_index", "section_index", "page_index"]
    values: List[Any] = [meta_book, meta_chap, meta_sect, meta_page]

    for c in cols:
        if c in ("book_name", "chapter_index", "section_index", "page_index", "id", "rowid"):
            continue
        if c.endswith("_has_data"):
            base = c[: -len("_has_data")]
            vec = tag_vectors.get(base)
            values.append(1 if vec is not None else 0)
            col_names.append(c)
            continue
        vec = tag_vectors.get(c)
        if vec is None:
            zero = np.zeros((EMBEDDING_DIMENSION,), dtype=np.float32)
            values.append(_to_f32_blob(zero))
        else:
            values.append(_to_f32_blob(vec))
        col_names.append(c)

    placeholders = ", ".join(["?"] * len(col_names))
    col_list = ", ".join(col_names)

    cur.execute(
        f"INSERT INTO vec_joined ({col_list}) VALUES ({placeholders})"
        , values,
    )
    rowid = cur.lastrowid
    return rowid


def search_by_tag_key(
    conn: sqlite3.Connection,
    query_vector: np.ndarray,
    tag_key: str,
    k: int = 5,
    distance_threshold: float = 0.99,
) -> List[Dict[str, Any]]:
    """Search nearest sections by a tag key using the single joined vec table.

    Args:
        conn: Database connection.
        query_vector: Query embedding.
        tag_key: Tag key to search (e.g., 'formulas', 'syndromes'). Must match a column in vec_joined.
        k: Max results to return.

    Returns:
        List of section metadata dicts with similarity scores.
    """
    cur = conn.cursor()

    cols = set(_get_table_columns(conn))
    if tag_key not in cols:
        return []

    qblob = _to_f32_blob(query_vector)

    cur.execute(
        f"""
        SELECT rowid, distance, book_name, chapter_index, section_index, page_index
        FROM vec_joined
        WHERE {tag_key}_has_data = 1
          AND {tag_key} MATCH ?
          AND k = ?
        ORDER BY distance ASC
        """,
        (qblob, k),
    )
    rows = cur.fetchall()

    results: List[Dict[str, Any]] = []
    for row in rows:
        rowid, distance, book_name, chapter_index, section_index, page_index = row
        results.append(
            {
                "id": rowid,
                "book_name": book_name,
                "chapter_index": chapter_index,
                "section_index": section_index,
                "page_index": page_index,
                "similarity": float(1.0 / (1.0 + float(distance))),
            }
        )

    return results


def get_section_columns(conn: sqlite3.Connection) -> List[str]:
    """Return column names of the joined vec table (metadata + vector columns)."""
    return _get_table_columns(conn)
