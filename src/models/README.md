# yQi RAG System

This folder contains the Retrieval-Augmented Generation (RAG) system for the yQi evaluation platform, enabling AI responses based on local Traditional Chinese Medicine (TCM) documents.

## Overview

The RAG system provides an alternative to ChatGPT by using local TCM documents as a knowledge base. It processes documents from the `docs/` folder, creates vector embeddings using OpenAI's API, and generates contextually relevant responses using retrieved document chunks.

## Components

### Core Files
- `rag_system.py` - Main RAG implementation with vector database and query processing
- `document_processor.py` - Document ingestion and text chunking for .doc/.docx files
- `__init__.py` - Module initialization and exports
- `requirements.txt` - Python dependencies
- `setup.py` - Automated setup script for dependencies and API configuration

### Key Features
- **Document Processing**: Supports .doc, .docx, and .txt files
- **Vector Database**: Local pickle-based storage with cosine similarity search
- **OpenAI Integration**: Uses OpenAI API for embeddings and text generation
- **TCM-Optimized**: Specialized prompts for Traditional Chinese Medicine evaluation

## Setup

### Quick Setup
Run the automated setup script:
```bash
cd models
python setup.py
```

### Manual Setup

1. **Install Python Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure OpenAI API**:
   - Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```
   - Or add it to your `.env` file in the evaluation directory

3. **Add Documents**:
   - Place your TCM documents (.doc, .docx, .txt) in the `../docs/` folder
   - The system will automatically process them on first use

## Usage

### In the Evaluation Platform
1. Start the evaluation platform: `streamlit run evaluation/app.py`
2. Select "RAG (Local Documents)" from the model dropdown
3. Configure your prompts as usual
4. Click "Generate RAG Responses"

### Programmatic Usage
```python
from models import RAGSystem, DocumentProcessor

# Initialize components
rag_system = RAGSystem()
doc_processor = DocumentProcessor()

# Build knowledge base (first time only)
chunks = doc_processor.process_documents()
rag_system.build_database_from_chunks(chunks)
rag_system.save_database()

# Query the system
result = rag_system.query("What herbs are used for kidney yang deficiency?")
print(result['response'])
```

## Configuration

### Default Models
- **Embedding Model**: `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf`
- **Language Model**: `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF`

### Customization
You can customize the models by modifying the RAGSystem initialization:
```python
rag_system = RAGSystem(
    embedding_model="your-embedding-model",
    language_model="your-language-model"
)
```

## Document Processing

### Supported Formats
- `.doc` - Microsoft Word (legacy)
- `.docx` - Microsoft Word (modern)
- `.txt` - Plain text files

### Processing Pipeline
1. **Text Extraction**: Extracts text from documents using python-docx or textract
2. **Chunking**: Splits text into overlapping chunks (default: 500 chars, 50 char overlap)
3. **Embedding**: Creates vector embeddings for each chunk
4. **Storage**: Saves to local pickle database

### Chunk Metadata
Each chunk includes:
- Source document name
- Chunk index and total chunks
- File path
- Processing timestamp

## Vector Database

### Storage
- **Format**: Pickle file (`vector_db.pkl`)
- **Location**: `models/` directory
- **Structure**: List of (text, embedding, metadata) tuples

### Retrieval
- **Method**: Cosine similarity search
- **Default**: Top 3 most relevant chunks
- **Configurable**: Adjust `top_n` parameter in queries

## System Prompts

### Default TCM Prompt
The system uses a specialized prompt for TCM evaluation:
- Focuses on diagnostic analysis and syndrome differentiation (辨證論治)
- Emphasizes herbal prescriptions with dosages
- Requires responses based strictly on provided context

### Custom Prompts
You can provide custom system prompts when querying:
```python
result = rag_system.query(
    "Your question here",
    system_prompt="Your custom system prompt"
)
```

## Troubleshooting

### Common Issues

**"Ollama not available"**
- Ensure Ollama is installed and running: `ollama serve`
- Verify models are installed: `ollama list`

**"No documents found"**
- Check that documents exist in `../docs/` folder
- Verify file extensions are supported (.doc, .docx, .txt)

**"Failed to build vector database"**
- Check Ollama models are properly installed
- Ensure sufficient disk space for embeddings
- Verify document text extraction is working

**Import errors**
- Install missing dependencies: `pip install -r requirements.txt`
- For .doc files, may need additional system dependencies for textract

### Performance Tips
- **First Run**: Building the vector database takes time (depends on document size)
- **Subsequent Runs**: Database is cached and loads quickly
- **Memory Usage**: Large document collections require more RAM for embeddings

## Integration with Evaluation Platform

The RAG system integrates seamlessly with the existing yQi evaluation platform:
- **File Compatibility**: Uses same response/benchmark JSON format
- **Rating System**: RAG responses can be rated alongside ChatGPT responses
- **Statistics**: Included in platform statistics and comparisons
- **Batch Processing**: Currently real-time only (no batch support)

## Development

### Adding New Document Types
Extend `DocumentProcessor.extract_text_from_doc()` to support additional formats.

### Custom Embedding Models
Modify `RAGSystem.get_embedding()` to use different embedding providers.

### Alternative Vector Stores
Replace pickle storage with databases like Chroma, Pinecone, or FAISS by modifying the database methods in `RAGSystem`.

## Dependencies

### Python Packages
- `ollama>=0.1.0` - Ollama Python client
- `python-docx>=0.8.11` - .docx file processing
- `textract>=1.6.5` - .doc file processing (fallback)
- `numpy>=1.21.0` - Numerical operations

### System Requirements
- **Ollama**: Local LLM runtime
- **Python**: 3.8 or higher
- **Memory**: 4GB+ RAM recommended for embeddings
- **Storage**: Space for documents + vector database
