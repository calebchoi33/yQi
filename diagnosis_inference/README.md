# TCM Diagnosis Inference Router

This module serves as the main entry point for Traditional Chinese Medicine (TCM) diagnosis inference, routing requests to appropriate workflows based on configuration and requirements.

## Configuration File Format

The configuration file (`config_example.json`) uses the following structure:

### Fields

#### `workflow` (string, required)
Specifies which workflow type to use for processing diagnosis requests.

**Options:**
- `"no_rag"` - Direct LLM approach using OpenAI API without retrieval
- `"chunk_rag"` - Chunk-based RAG using vector similarity search
- `"tag_rag"` - Structured tag-based RAG with multi-vector search

#### `prompts` (array, required)
List of patient cases/prompts to process through the selected workflow. Each prompt object contains:

- **`id`** (string) - Unique identifier for the case (e.g., "case_001")
- **`description`** (string) - Brief description of the patient condition
- **`content`** (string) - Full patient case description including symptoms, tongue diagnosis, pulse diagnosis, and any other relevant TCM diagnostic information

### Example Configuration

```json
{
  "workflow": "chunk_rag",
  "prompts": [
    {
      "id": "case_001",
      "description": "Patient with digestive issues",
      "content": "A 45-year-old male patient presents with chronic stomach pain..."
    }
  ]
}
```

## Workflow Types

### No-RAG Workflow
- Uses direct OpenAI API calls without document retrieval
- Requires `OPENAI_API_KEY` environment variable
- Best for general TCM knowledge without specific document context

### Chunk-RAG Workflow  
- Uses vector-based document retrieval with chunking
- Retrieves relevant document sections based on similarity
- Combines retrieved context with LLM generation

### Tag-RAG Workflow
- Uses structured vector database with tag-based organization
- Supports multi-vector search across different document sections
- More sophisticated retrieval for complex TCM knowledge

## Usage

```bash
python give_diagnosis.py --config config_example.json --output results.json
```

For single case processing:
```bash
python give_diagnosis.py --config config_example.json --case "Patient description..."
```
