# yQi AI Evaluation Platform

A comprehensive AI evaluation platform for testing and rating LLM responses with both real-time and batch processing capabilities.

## Quick Start

1. **Clone the Project**
   ```bash
   git clone git@github.com:calebchoi33/yQi.git
   cd yQi
   ```

2. **Install dependencies:**
   ```bash
   cd evaluation
   python -m venv myenv
   pip install -r requirements.txt
   ```
   To run your virtual environment (venv) in the future
   ```bash
   source myenv/bin/activate
   ```

4. **Set up OpenAI API key:**
   ```bash
   nano .env
   # Add to .env file or enter in app sidebar
   OPENAI_API_KEY="your-api-key-here"
   ```
   If one does not exist
   ```bash
   touch .env
   ```
6. **Run the app:**
   ```bash
   streamlit run app.py
   ```

7. **Open browser:** Navigate to `http://localhost:8501`

## How to Use

### Generate Responses
1. **Choose Processing Mode:**
   - **Real-time**: Immediate responses (5 prompts at a time)
   - **Batch**: Cost-effective bulk processing (up to 50,000 prompts)

2. **Configure Settings:**
   - Enter system prompt or use defaults
   - More to choose from with the usage of additional models

3. **Submit Prompts:**
   - Use default prompts or upload custom ones
   - Real-time: Results appear immediately
   - Batch: Monitor job status and download when complete (takes a very long time, but is cheap)

### Rate Responses
1. **Select Run:** Choose from completed evaluation runs
2. **Rate Each Response:** Use 1-5 scale with comments
3. **Track Progress:** See completion status and statistics

### View Results
- **Responses Tab:** Browse all generated responses
- **Statistics Tab:** View performance metrics and ratings
- **Files:** All data saved in `responses/` and `benchmarks/` folders

## Key Features

- **Dual Processing Modes:** Real-time for quick tests, batch for large-scale evaluations
- **Cost Optimization:** Batch processing offers 50% cost savings
- **Persistent Storage:** All data saved locally in organized date folders
- **Rating System:** Comprehensive evaluation with statistics tracking
- **Job Management:** Monitor long-running batch jobs across app restarts

## Architecture

```
evaluation/
├── app.py                 # Main Streamlit application
├── api_client.py          # OpenAI API integration
├── batch_manager.py       # Batch job lifecycle management
├── file_manager.py        # Data storage and retrieval
├── rating_manager.py      # Rating system and statistics
├── ui/                    # User interface components
├── responses/             # Generated responses (by date)
├── benchmarks/            # Performance metrics (by date)
├── batches/               # Batch job files (by date)
└── ratings/               # User ratings (by run)
```

## Requirements

- Python 3.8+
- OpenAI API key with GPT-4o access
- Internet connection for API calls

## Data Storage

All data is stored locally:
- **Responses:** `responses/YYYY-MM-DD/responses_*.json`
- **Benchmarks:** `benchmarks/YYYY-MM-DD/benchmark_*.json`
- **Ratings:** `ratings/run_id/rating_*.json`
- **Batch Jobs:** `batch_jobs.json` (persistent tracking)
