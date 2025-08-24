# yQi Evaluation Platform

A streamlined Streamlit application for evaluating Traditional Chinese Medicine (TCM) AI capabilities using ChatGPT-4o. The platform features async batch processing, local JSON file storage, and a comprehensive rating system for systematic evaluation of AI responses.

## Features

### Core Evaluation
- Comprehensive TCM medical case evaluation prompts
- Uses GPT-4o model for enhanced diagnostic reasoning
- **Async batch processing** for improved performance
- Real-time progress tracking during API calls
- Organized file storage in separate `responses/` and `benchmarks/` folders
- Comprehensive benchmarking with performance metrics

### Rating System
- **Advanced rating interface** with multiple criteria
- **Statistics and analytics** for rating data
- Multi-rater support with attribution
- **JSON file storage** for ratings and responses
- Response filtering and navigation capabilities

### Technical Features
- Beautiful, organized display of responses with TCM analysis
- Download responses and benchmark data as JSON files
- Secure API key handling (environment variable or secure input)
- Detailed timing and token usage analytics
- Robust retry logic for handling API rate limits and connection errors
- Detailed logging for troubleshooting and debugging

## Installation

1. Clone or download this project
2. Navigate to the project directory:
   ```bash
   cd evaluation
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Setup

### OpenAI API Key

You need an OpenAI API key to use this application. You can obtain one from [OpenAI's website](https://platform.openai.com/api-keys).

**Option 1: Environment Variable (Recommended)**
```bash
# Add to your .env file
OPENAI_API_KEY="your-api-key-here"
```

**Option 2: Enter in the App**
The app will prompt you to enter your API key in the sidebar if it's not found in the environment.

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the provided URL (usually `http://localhost:8501`)

3. **Generate Responses Tab:**
   - Choose to use default prompts or enter custom ones
   - Enter your OpenAI API key if not set as environment variable
   - Click "Generate Responses" to send prompts to ChatGPT
   - Watch the progress bar as responses are collected with async batch processing
   - Data is automatically saved to JSON files

4. **View Responses Tab:**
   - View all saved responses in an organized format
   - See summary metrics (total prompts, tokens used, generation date)
   - Download the JSON file with all responses

5. **Rate Responses Tab:**
   - Select an evaluation run to rate
   - Navigate through responses using Previous/Next buttons
   - Rate responses on multiple criteria (1-5 scale):
     - Overall Quality
     - Medical Accuracy
     - Completeness
     - Clarity
     - Clinical Relevance
     - Safety
   - Add optional comments for detailed feedback
   - Track rating progress with visual indicators

6. **Statistics Tab:**
   - View rating statistics and analytics
   - See average ratings by criteria
   - Analyze rating distributions
   - Compare performance across evaluation runs

## Default Medical Cases

The application comes with 20 default Chinese medical case prompts covering:
1. **Respiratory conditions** - fever, cough, pneumonia symptoms
2. **Gastrointestinal issues** - digestive problems, diarrhea, constipation
3. **Cardiovascular symptoms** - palpitations, chest tightness
4. **Allergies and skin conditions** - seasonal allergies, dermatitis
5. **Urological problems** - kidney stones, UTI symptoms
6. **Pediatric cases** - ADHD, Tourette's, bedwetting
7. **Gynecological conditions** - menstrual irregularities, menopause
8. **Mental health cases** - depression, anxiety, OCD
9. **Complex conditions** - cancer recovery, genetic disorders

Each prompt requests:
- **Chain of thought analysis**: TCM diagnostic reasoning including symptom analysis and syndrome differentiation
- **Chinese herbal prescriptions**: Specific herbs and dosages for treatment

## Application Structure

The application has been refactored into a modular structure for better maintainability and separation of concerns:

### Core Application
- **app.py** - Main Streamlit application entry point with 4-tab interface
- **api_client.py** - Async OpenAI API interaction with batch processing
- **file_manager.py** - File operations for saving/loading responses and benchmarks
- **rating_manager.py** - File-based rating system for response evaluation
- **simple_rating_interface.py** - Streamlined rating interface
- **prompts.py** - Default TCM evaluation prompts

### Core Modules
- **core/config.py** - Centralized configuration management
- **core/services.py** - Service factory and management
- **core/exceptions.py** - Custom exception classes

### UI Components
- **ui/components.py** - Reusable Streamlit UI components
- **ui/tabs.py** - Tab management and rendering

## File Storage

The application creates organized directories for data storage:

### Responses Directory (`responses/`)
Stores ChatGPT responses in timestamped files: `responses_YYYYMMDD_HHMMSS.json`

```json
{
  "timestamp": "2025-08-20T00:30:00.123456",
  "total_prompts": 20,
  "responses": [
    {
      "prompt_number": 1,
      "prompt": "病人，男，25歲，亞裔...",
      "response": "中醫診斷分析: ...",
      "tokens_used": 450,
      "timestamp": "2025-08-20T00:30:05.123456"
    }
  ]
}
```

### Benchmarks Directory (`benchmarks/`)
Stores performance metrics in timestamped files: `benchmark_YYYYMMDD_HHMMSS.json`

```json
{
  "run_id": "20250820_003000",
  "model": "gpt-4o",
  "total_duration_seconds": 45.2,
  "total_tokens_used": 8500,
  "success_rate": 1.0,
  "tokens_per_second": 188.1,
  "prompt_processing_times": [...]
}
```

## Requirements

### System Requirements
- Python 3.8+
- Valid OpenAI API key with GPT-4o access

### Python Dependencies
- Streamlit 1.32.0
- OpenAI Python library 1.6.1
- python-dotenv 1.0.0
- pandas 2.1.4

## File Storage Structure

The application uses JSON files for all data storage:

- **responses/** - Individual AI responses with metadata
- **benchmarks/** - Performance metrics for each evaluation run
- **ratings/** - User ratings organized by evaluation run
- All files are timestamped for easy organization and retrieval

## Performance Improvements

### Async Batch Processing
The platform now uses async batch processing to significantly improve performance:

- **Before (Sequential)**: ~201.89 seconds for 20 prompts (10.09s avg)
- **After (Async Batches)**: ~178.88 seconds for 20 prompts (8.94s avg)
- **Improvement**: ~11.4% faster processing time

### Benefits
- Concurrent API calls within batches
- Better resource utilization
- Improved error handling and retry logic
- Real-time progress tracking

## Troubleshooting

### File Permission Issues
```bash
# Ensure proper permissions for data directories
chmod 755 responses/ benchmarks/ ratings/
```

### Performance Issues
- Adjust batch size in core configuration for optimal performance
- Monitor API rate limits in the logs
- Use smaller prompt sets for testing

### Storage Issues
- Check available disk space for JSON file storage
- Clean up old evaluation runs if needed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.
