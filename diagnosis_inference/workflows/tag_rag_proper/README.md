# Tag RAG Proper System (SQLite + sqlite-vec)

A clean, efficient tag-based Retrieval-Augmented Generation (RAG) system for Traditional Chinese Medicine (TCM) knowledge retrieval using SQLite with vector search (sqlite-vec).

## Overview

This system implements a structured approach to semantic search over TCM clinical data by:

1. **Discovering tag keys** from the JSON sections (e.g., formulas, syndromes, treatments) at schema setup time
2. **Creating a single sqlite-vec table** with multiple vector columns (one per discovered tag key) and metadata columns
3. **Enabling targeted search** by matching against a specific vector column
4. **Using normalized embeddings** for consistent similarity scoring

## Architecture

### Core Components

- `database.py`: SQLite database with vector search (sqlite-vec) and a simple schema
- `embeddings.py`: OpenAI embedding utilities and tag processing helpers
- `ingestion.py`: Data pipeline for processing and writing vectors/metadata
- `query_engine.py`: Query and RAG pipeline (retrieval + generation)
- `test_system.py`: Smoke tests for ingestion, structure, and querying

### Database Schema

The schema is intentionally minimal and fast:

- Single sqlite-vec virtual table `vec_joined` created via `USING vec0(...)` that includes:
  - Metadata columns: `book_name` (TEXT), `chapter_index` (TEXT), `section_index` (TEXT), `page_index` (INTEGER)
  - One vector column per discovered tag key (e.g., `formulas float[1536]`, `symptoms float[1536]`, ...)
  - If a section has no tags for a given key, that vector column is stored as NULL

## Setup

### 1) Install Dependencies

```bash
pip install -r requirements.txt
```

### 2) Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3) sqlite-vec

The Python package `sqlite-vec` is included in `requirements.txt` and registers the vector extension automatically upon import.

## Usage

### Data Ingestion

Process and store the tag data into the single joined table:

```python
from ingestion import ingest_all_sections

# Requires OPENAI_API_KEY in the environment
count = ingest_all_sections()
print(f"Ingested {count} sections")
```

### RAG Pipeline (Retrieval + Generation)

Complete diagnosis in one call:

```python
from query_engine import diagnose

result = diagnose(
    patient_case="患者發熱惡寒，頭痛身痛...",
    tag_keys=["symptoms", "syndromes", "formulas"],
    k=3
)

print(result["diagnosis"])          # LLM-generated diagnosis
print(result["formatted_context"])  # Retrieved TCM knowledge
```

### Lower-Level Query Functions

For retrieval only (without LLM generation):

```python
from query_engine import query, multi_key_query

results = query("fever and headache", "symptoms", k=5)
multi = multi_key_query("大陷胸湯", ["formulas", "treatments"], k=3)
```

## Tag Keys

Tag keys are discovered from the JSON data at schema setup. Common keys: formulas, syndromes, treatments, symptoms, organs, herbs, pulses, acupoints, meridians, elements, tongues, pathogens.

The system creates vector columns inside `vec_joined` for each discovered tag key.

## Testing

Run the smoke tests:

```bash
python test_system.py
```

This will:
1) Test data ingestion pipeline
2) Verify database structure (single `vec_joined` table with metadata + per-tag vector columns)
3) Test query functionality across discovered tag keys

## Configuration

Key settings in `embeddings.py`:

- **EMBEDDING_MODEL**: `text-embedding-3-small` (1536 dimensions)
- **EMBEDDING_DIMENSION**: 1536
- **TAGS_JSON_PATH**: Path to the tagged TCM text JSON file
- **DEFAULT_TOP_K**: Default number of results (5)

## Key Features

- Dynamic tag key discovery from JSON (no hardcoded schemas)
- Single `vec_joined` table with multiple vector columns for targeted search
- Normalized embeddings for consistent similarity scoring
- Complete RAG pipeline: `diagnose()` handles retrieval + LLM generation
- NULL handling for sparse tag data

## Data Source

The system processes `《人紀傷寒論》_tags.json` which contains:
- TCM clinical sections with metadata (book_name, chapter_idx, section_idx)
- Tag arrays per section (e.g., formulas, syndromes, symptoms)
- Bilingual tags with `name_zh` and `name_en` fields

The system discovers which tag keys exist in the data and creates corresponding vector columns in `vec_joined` accordingly.
