# Vector RAG System

Traditional Chinese Medicine (TCM) Retrieval-Augmented Generation system using vector embeddings for semantic search and bilingual response generation.

## Overview

This subproject implements a RAG system that:
- Processes TCM documents with various chunking strategies
- Creates vector embeddings using OpenAI's embedding models
- Performs semantic search to retrieve relevant content
- Generates bilingual (Chinese/English) diagnostic responses

## Key Components

- `rag_system.py` - Core RAG implementation with OpenAI integration
- `enhanced_rag_system.py` - Enhanced version with improved retrieval
- `document_processor.py` - Document parsing and chunking
- `chunking_strategies.py` - Various text chunking methods
- `text_extractors.py` - Extract text from different file formats

## Configuration Files

- `eval_config.json` - Basic evaluation configuration
- `eval_config_semantic.json` - Semantic chunking configuration

## How to Run

1. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

2. Run evaluation:
```bash
cd vector_rag
python run_eval.py --config eval_config_semantic.json
```

3. Start the evaluation UI:
```bash
cd vector_rag/evaluation
streamlit run app.py
```

## Features

- Multiple chunking strategies (regular, chapter, section, semantic)
- Bilingual response generation
- Interactive evaluation interface
- Batch evaluation with metrics
- Support for .txt, .doc, .docx files
