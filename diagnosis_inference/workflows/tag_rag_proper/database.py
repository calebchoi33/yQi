"""PostgreSQL database with pgvector for Tag RAG system."""

import psycopg2
import psycopg2.extras
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

from embeddings import TAG_FAMILIES, EMBEDDING_DIMENSION, DB_NAME, DB_HOST, DB_PORT, DB_USER, DB_PASSWORD

logger = logging.getLogger(__name__)

def setup_database() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    
    create_schema(conn)
    create_vector_indexes(conn)
    return conn

def create_schema(conn: psycopg2.extensions.connection):
    # Create main sections table
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS sections (
        id SERIAL PRIMARY KEY,
        book_name TEXT,
        chapter_index TEXT,
        section_index TEXT,
        page_index INTEGER,
        
        -- Vector columns for all tag families
        formulas_vec VECTOR({EMBEDDING_DIMENSION}),
        syndromes_vec VECTOR({EMBEDDING_DIMENSION}),
        treatments_vec VECTOR({EMBEDDING_DIMENSION}),
        pathogens_vec VECTOR({EMBEDDING_DIMENSION}),
        organs_vec VECTOR({EMBEDDING_DIMENSION}),
        herbs_vec VECTOR({EMBEDDING_DIMENSION}),
        symptoms_vec VECTOR({EMBEDDING_DIMENSION}),
        pulses_vec VECTOR({EMBEDDING_DIMENSION}),
        acupoints_vec VECTOR({EMBEDDING_DIMENSION}),
        meridians_vec VECTOR({EMBEDDING_DIMENSION}),
        elements_vec VECTOR({EMBEDDING_DIMENSION}),
        tongues_vec VECTOR({EMBEDDING_DIMENSION}),
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    with conn.cursor() as cur:
        cur.execute(create_table_sql)
    conn.commit()

def create_vector_indexes(conn: psycopg2.extensions.connection):
    with conn.cursor() as cur:
        for family_name, family_config in TAG_FAMILIES.items():
            vector_column = family_config["vector_column"]
            index_name = f"idx_{vector_column}_hnsw"
            
            create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS {index_name} 
            ON sections USING hnsw ({vector_column} vector_cosine_ops)
            """
            cur.execute(create_index_sql)
    
    conn.commit()

def insert_section(conn: psycopg2.extensions.connection, section_data: Dict[str, Any], tag_vectors: Dict[str, np.ndarray]) -> int:
    vector_columns = [TAG_FAMILIES[family]["vector_column"] for family in TAG_FAMILIES.keys()]
    columns_str = "book_name, chapter_index, section_index, page_index, " + ", ".join(vector_columns)
    placeholders = ", ".join(["%s"] * (4 + len(vector_columns)))
    
    insert_sql = f"""
    INSERT INTO sections ({columns_str})
    VALUES ({placeholders})
    RETURNING id
    """
    
    vector_values = []
    for family in TAG_FAMILIES.keys():
        vector = tag_vectors.get(family)
        vector_values.append(vector.tolist() if vector is not None else None)
    
    with conn.cursor() as cur:
        cur.execute(insert_sql, (
            section_data.get("book_name", ""),
            section_data.get("chapter_idx", ""),
            section_data.get("section_idx", ""),
            section_data.get("page_index", 0),
            *vector_values
        ))
        section_id = cur.fetchone()[0]
    
    conn.commit()
    return section_id


def search_by_tag_family(conn: psycopg2.extensions.connection, query_vector: np.ndarray, tag_family: str, k: int = 5) -> List[Dict[str, Any]]:
    if tag_family not in TAG_FAMILIES:
        raise ValueError(f"Unknown tag family: {tag_family}")
    
    vector_column = TAG_FAMILIES[tag_family]["vector_column"]
    
    search_sql = f"""
    SELECT 
        id,
        book_name,
        chapter_index,
        section_index,
        page_index,
        1 - ({vector_column} <=> %s::vector) as similarity
    FROM sections
    WHERE {vector_column} IS NOT NULL
    ORDER BY {vector_column} <=> %s::vector
    LIMIT %s
    """
    
    with conn.cursor() as cur:
        query_list = query_vector.tolist()
        cur.execute(search_sql, (query_list, query_list, k))
        results = []
        for row in cur.fetchall():
            results.append({
                "id": row[0],
                "book_name": row[1],
                "chapter_index": row[2],
                "section_index": row[3],
                "page_index": row[4],
                "similarity": row[5]
            })
        return results

def get_section_columns(conn: psycopg2.extensions.connection) -> List[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'sections'")
        return [row[0] for row in cur.fetchall()]
