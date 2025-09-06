# yQi - Traditional Chinese Medicine RAG System

A Retrieval-Augmented Generation (RAG) system for Traditional Chinese Medicine (TCM) documents with enhanced document processing and evaluation capabilities.

## 📁 Project Structure

```
yQi/
├── config/                    # Configuration files
│   ├── create_vdb_config.json # Vector database creation settings
│   ├── eval_config.json       # Evaluation configuration
│   └── patient_cases.json     # TCM patient test cases
├── scripts/                   # Main execution scripts
│   ├── create_new_vdb.py      # Vector database creation
│   └── run_eval.py            # Command-line evaluation runner
├── tests/                     # Test and debug scripts
│   ├── test_chunking.py       # Chunking functionality tests
│   ├── test_enhanced_rag.py   # RAG system integration tests
│   └── test_refactored_processor.py # Document processor tests
├── output/                    # Generated output files
│   ├── out.json              # Legacy output
│   ├── out_enhanced.json     # Enhanced output
│   └── test_results.json     # Test results
├── models/                    # Core system modules
│   ├── rag_system.py         # Main RAG system
│   ├── document_processor.py # Document processing with TCM chunking
│   ├── text_extractors.py    # Text extraction strategies
│   └── chunking_strategies.py # Modular chunking methods
├── docs/                      # TCM documents
│   └── 倪海廈人紀電子書 - 傷寒論.txt
├── evaluation/                # Evaluation platform
│   ├── app.py                # Streamlit evaluation UI
│   └── benchmarks/           # Evaluation results
└── run_eval.py               # Convenience wrapper (backward compatibility)
└── create_new_vdb.py         # Convenience wrapper (backward compatibility)
```

## 🚀 Quick Start

### 1. Create Vector Database
```bash
# Using convenience wrapper
python3 create_new_vdb.py --config config/create_vdb_config.json

### 2. Run Evaluation
```bash
# Using convenience wrapper  
python3 run_eval.py --config config/eval_config.json --cases config/patient_cases.json

### 3. Launch Evaluation UI
```bash
cd evaluation
streamlit run app.py
```

## 🔧 Features

### Enhanced Document Processing
- **Multi-format support**: `.txt`, `.docx`, `.doc`, `.pdf`
- **TCM-aware chunking**: Chapter, section, and formula-based chunking
- **Modular architecture**: Pluggable text extractors and chunking strategies
- **Metadata enrichment**: Chapter titles, section titles, chunk types

### RAG System
- **OpenAI Integration**: Uses `text-embedding-3-small` for embeddings, `gpt-4o-mini` for generation
- **Vector Database**: Pickle-based storage with cosine similarity search
- **Flexible Retrieval**: Configurable top-k retrieval with metadata filtering

### Evaluation Platform
- **Streamlit UI**: Interactive evaluation interface
- **Batch Processing**: Command-line evaluation for automation
- **Result Tracking**: JSON export with timestamps and metadata
- **Model Comparison**: Support for multiple RAG configurations

## 📋 Configuration

### Vector Database Config (`config/create_vdb_config.json`)
```json
{
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "chunking_method": "section",
  "detect_tcm_markers": true
}
```

### Evaluation Config (`config/eval_config.json`)
```json
{
  "vector_db_path": "models/tcm_vector_db.pkl",
  "top_k": 3,
  "system_prompt": "You are a TCM expert..."
}
```

## 🧪 Testing

Run individual test suites:
```bash
python3 tests/test_refactored_processor.py
python3 tests/test_enhanced_rag.py  
python3 tests/test_chunking.py
```

## 📊 Output

- **Evaluation results**: `evaluation/benchmarks/` and `evaluation/responses/`
- **Test outputs**: `output/`
- **Vector databases**: `models/`

## 🔄 Migration Notes

This project has been reorganized for better maintainability:
- Configuration files moved to `config/`
- Scripts organized in `scripts/` with root-level wrappers for compatibility
- Tests consolidated in `tests/`
- Output files organized in `output/`

All existing workflows remain compatible through convenience wrapper scripts.
