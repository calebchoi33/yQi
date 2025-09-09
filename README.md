# yQi - Traditional Chinese Medicine RAG System

A Retrieval-Augmented Generation (RAG) system for Traditional Chinese Medicine (TCM) documents with enhanced document processing and evaluation capabilities.

## 📁 Project Structure

```
yQi/
├── config/                    # Configuration files
│   ├── create_vdb_config.json # Vector database creation settings
│   ├── eval_config.json       # Evaluation configuration
│   ├── eval_config_semantic.json # Semantic evaluation config
│   └── patient_cases.json     # TCM patient test cases
├── src/                       # Source code
│   ├── scripts/               # Main execution scripts
│   │   ├── create_new_vdb.py  # Vector database creation
│   │   ├── create_semantic_vdb.py # Semantic vector database
│   │   └── run_eval.py        # Command-line evaluation runner
│   └── models/                # Core system modules
│       ├── rag_system.py      # Main RAG system with bilingual support
│       ├── document_processor.py # Document processing with TCM chunking
│       ├── text_extractors.py # Text extraction strategies
│       └── chunking_strategies.py # Modular chunking methods
├── data/                      # Data storage
│   ├── documents/             # TCM source documents
│   │   └── 倪海廈人紀電子書 - 傷寒論.txt
│   └── vector_dbs/            # Vector databases
│       ├── semantic_vector_db.pkl
│       └── vector_db_enhanced.pkl
├── tests/                     # Test and debug scripts
│   ├── test_chunking.py       # Chunking functionality tests
│   ├── test_enhanced_rag.py   # RAG system integration tests
│   └── test_refactored_processor.py # Document processor tests
├── output/                    # Generated output files
├── evaluation/                # Evaluation platform
│   ├── app.py                # Streamlit evaluation UI
│   ├── prompts.py            # Default evaluation prompts
│   ├── results/              # Evaluation results
│   ├── benchmarks/           # Performance benchmarks
│   └── mock_retrieval_data.json # Mock data for testing
├── run_eval.py               # Convenience wrapper
├── create_vdb.py             # Convenience wrapper
└── create_semantic_vdb.py    # Convenience wrapper
```

## 🚀 Quick Start

### 1. Create Vector Database
```bash
# Create regular vector database
python3 create_vdb.py --config config/create_vdb_config.json

# Create semantic vector database (recommended)
python3 create_semantic_vdb.py --config config/create_vdb_config_semantic.json
```

### 2. Run Evaluation
```bash
# Run bilingual RAG evaluation
python3 run_eval.py --config config/eval_config_semantic.json --output evaluation/results/my_test.json

# Run with mock data for testing
python3 run_eval.py --config config/eval_config_semantic.json --output evaluation/results/mock_test.json
```

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
- **OpenAI Integration**: Uses `text-embedding-3-large` for embeddings, `gpt-4o-mini` for generation
- **Bilingual Support**: Generates responses in both Chinese and English
- **Vector Database**: Pickle-based storage with cosine similarity search
- **Adjacent Chunk Retrieval**: Enhanced context with neighboring chunks
- **Mock Mode**: Testing capability with predefined retrieval data
- **Translated Chunks**: Automatic translation of retrieved Chinese text to English

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

### Evaluation Config (`config/eval_config_semantic.json`)
```json
{
  "use_rag": true,
  "docs_directory": "data/documents",
  "vector_db_path": "data/vector_dbs/semantic_vector_db.pkl",
  "model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-large",
  "top_n_chunks": 3,
  "chunking_method": "semantic_section",
  "preserve_semantic_boundaries": true
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

- **Evaluation results**: `evaluation/results/` (consolidated JSON with bilingual responses)
- **Performance benchmarks**: `evaluation/benchmarks/`
- **Response archives**: `evaluation/responses/`
- **Vector databases**: `data/vector_dbs/`
- **Test outputs**: `output/`

## 🔄 Recent Updates

### Enhanced Features
- **Bilingual RAG**: Responses generated in both Chinese and English
- **Translated Retrieval**: Chinese chunks automatically translated to English
- **Adjacent Chunks**: Enhanced context with neighboring document sections
- **Mock Testing**: Controlled testing with predefined retrieval data
- **Timestamped Output**: Automatic file naming with date/time stamps

### Refactored Structure
- **Cleaner Organization**: Source code moved to `src/`, data to `data/`
- **Deprecated Files Removed**: Eliminated backup files and redundant scripts
- **Updated Paths**: All configurations updated for new folder structure
- **Convenience Wrappers**: Root-level scripts maintain backward compatibility

All existing workflows remain compatible through convenience wrapper scripts.
