# Structured RAG System

Advanced TCM RAG system using structured vector database with multi-vector columns for specialized semantic search across different aspects of TCM knowledge.

## Overview

This subproject implements a sophisticated RAG system that:
- Organizes TCM content by chapters and sections in a structured database
- Creates multiple specialized vector embeddings per record (symptoms, organs, formulas, patterns)
- Enables weighted multi-vector search for precise retrieval
- Supports adjacent content expansion for richer context
- Generates bilingual TCM diagnostic responses

## Key Components

- `structured_vector_db.py` - Multi-vector structured database implementation
- `structured_rag_system.py` - RAG system using structured database
- `create_vdb.py` - Script to build the structured vector database
- `structured_vdb_config.json` - Configuration for database schema and content

## Database Schema

Each record contains:
- **Vector Columns**: symptoms_vector, organs_vector, formulas_vector, patterns_vector, full_content_vector
- **Metadata**: book_name, chapter/section indices, content type, timestamps
- **Content**: Original text organized by chapter/section hierarchy

## How to Run

1. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

2. Create the structured vector database:
```bash
cd structured_rag
python create_vdb.py --config structured_vdb_config.json --output ../data/vector_dbs/structured_vector_db.pkl
```

3. Test the system:
```bash
python test_structured_rag.py
```

## Features

- Multi-vector semantic search (symptoms, organs, formulas, syndrome patterns)
- Adjacent content retrieval (sections, pages, chapters)
- Weighted vector scoring for precise results
- Chapter/section-based content organization
- Bilingual response generation with function calling
- Configurable retrieval settings
