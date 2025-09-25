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

### Prerequisites
- Python 3.8+
- OpenAI API key
- Required packages: `pip install openai streamlit python-dotenv`

### Environment Setup
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## 📊 Subprojects

### Vector RAG System
Standard RAG implementation with semantic search and bilingual responses.

**Features:**
- Multiple chunking strategies
- Vector embeddings with OpenAI
- Interactive evaluation UI
- Batch processing

**Usage:**
```bash
cd vector_rag
python run_eval.py --config eval_config_semantic.json
```

### Structured RAG System  
Advanced RAG with multi-vector structured database for specialized TCM knowledge retrieval.

**Features:**
- Multi-vector columns (symptoms, organs, formulas, patterns)
- Chapter/section organization
- Adjacent content expansion
- Weighted vector search

**Usage:**
```bash
cd structured_rag
python create_vdb.py --config structured_vdb_config.json
python test_structured_rag.py
```

## 🔧 Key Features

- **Bilingual Responses**: Chinese first, then English
- **Multiple RAG Approaches**: Standard vector search vs. structured multi-vector
- **TCM-Specific**: Optimized for Traditional Chinese Medicine content
- **Evaluation Tools**: Interactive UI and batch processing
- **Flexible Configuration**: JSON-based configuration system

## 📝 Getting Started

1. Choose your RAG approach (vector_rag or structured_rag)
2. Navigate to the subproject directory
3. Follow the README instructions for that subproject
4. Set your OpenAI API key
5. Run the evaluation or testing scripts

Each subproject is self-contained with its own documentation and can be run independently.

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
