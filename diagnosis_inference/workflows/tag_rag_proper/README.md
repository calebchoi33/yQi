# Tag RAG Proper System

A clean, efficient tag-based Retrieval-Augmented Generation (RAG) system for Traditional Chinese Medicine (TCM) knowledge retrieval using SQLite with vector search capabilities.

## Overview

This system implements a structured approach to semantic search over TCM clinical data by:

1. **Categorizing tags** into semantic families (symptoms, pulse patterns, general terms)
2. **Creating separate vector indexes** for each tag family
3. **Enabling targeted search** by specific medical domains
4. **Using normalized embeddings** for consistent similarity scoring

## Architecture

### Core Components

- **`config.py`**: Configuration settings including embedding model and tag families
- **`database.py`**: SQLite database with vector search capabilities
- **`embeddings.py`**: OpenAI embedding service and tag processing utilities
- **`ingestion.py`**: Data pipeline for processing and storing tag data
- **`query_engine.py`**: Query interface for semantic search
- **`test_system.py`**: Comprehensive test suite

### Database Schema

**sections table:**
- Metadata: `book_name`, `chapter_index`, `chapter_title`, `section_index`, `section_title`, `page_index`
- Tag JSON: `symptoms_json`, `pulse_json`, `general_json` (raw tag data)
- Vectors: `symptoms_vec`, `pulse_vec`, `general_vec` (normalized embeddings)

**Vector indexes:**
- `symptoms_index`: Vector index for symptom-related tags
- `pulse_index`: Vector index for pulse pattern tags  
- `general_index`: Vector index for general TCM terms

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Install SQLite Vector Extension

The system supports `sqlite-vss` for optimized vector search. If not available, it falls back to manual cosine similarity.

## Usage

### Data Ingestion

Process and store the tag data:

```python
from ingestion import DataIngestionPipeline

with DataIngestionPipeline("tag_rag.db") as pipeline:
    count = pipeline.ingest_all_sections()
    print(f"Ingested {count} sections")
```

### Querying

Search by specific tag family:

```python
from query_engine import TagRAGQueryEngine

with TagRAGQueryEngine("tag_rag.db") as engine:
    # Search symptoms
    results = engine.query("fever and headache", "symptoms", k=5)
    
    # Search pulse patterns
    results = engine.query("floating pulse", "pulse", k=3)
    
    # Multi-family search
    multi_results = engine.multi_family_query("floating pulse with fever", k=3)
```

### Query Response Format

Each result contains:
- `book_name`: Source book
- `chapter_index`: Chapter identifier
- `chapter_title`: Chapter name
- `section_index`: Section identifier  
- `section_title`: Section content
- `page_index`: Page number (if available)
- `tag_family`: Which semantic field matched
- `similarity_score`: Cosine similarity score

## Tag Families

The system categorizes tags into three families:

1. **symptoms**: Clinical manifestations, signs, symptoms
2. **pulse**: Pulse patterns and characteristics
3. **general**: General TCM terms, theories, concepts

## Testing

Run the comprehensive test suite:

```bash
python test_system.py
```

This will:
1. Test data ingestion pipeline
2. Verify database structure
3. Test query functionality across all tag families
4. Provide performance metrics

## Configuration

Key settings in `config.py`:

- **EMBEDDING_MODEL**: `text-embedding-3-small` (1536 dimensions)
- **TAG_FAMILIES**: Semantic categories for tag classification
- **DEFAULT_TOP_K**: Default number of results (5)
- **SIMILARITY_THRESHOLD**: Minimum similarity for results (0.7)

## Performance Features

- **Normalized embeddings** for consistent cosine similarity
- **Separate vector indexes** for targeted search
- **Batch embedding processing** for efficient ingestion
- **Fallback similarity search** when vector extensions unavailable
- **Connection pooling** and proper resource management

## Data Source

The system processes data from `../../tagging/tags.json` which contains:
- TCM clinical sections with metadata
- Bilingual tags (Chinese and English)
- Hierarchical book/chapter/section structure

This implementation follows the exact specifications provided, creating a clean and efficient tag-based RAG system optimized for TCM knowledge retrieval.
