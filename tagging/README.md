TCM Book Tagging
=================

Automatically tag sections of a Traditional Chinese Medicine (TCM) book into 12 categories using an LLM with tool calls. Tags are saved with per-chapter checkpoints so you can resume progress safely.

What it tags
------------
- symptom 症狀
- herb 中藥
- formula 方劑
- pulse 脈診
- tongue 舌診
- syndrome 證候
- pathogen 病因
- treatment 治法
- meridian 經絡
- organ 臟腑
- acupoint 穴位
- element 五行

Prerequisites
-------------
- Python 3.8+
- Packages: `openai`, `python-dotenv`
- OpenAI API key in environment or `.env`

Install dependencies:

```bash
pip install openai python-dotenv
```

Set environment (one of):

```bash
# Bash / Git Bash / WSL
export OPENAI_API_KEY="your-api-key"
# optional, defaults to gpt-4.1
export OPENAI_MODEL="gpt-4.1"
```

```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:OPENAI_MODEL = "gpt-4.1"
```

You can also create a `.env` file in this `tagging/` directory with:

```
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4.1
```

Prepare the book
----------------
Place your source text at `tagging/data/book.txt` formatted with chapter and section markers:

```text
#CHAPTER Chapter 1 Title
#SECTION Section 1 Title
<section 1 text>

#SECTION Section 2 Title
<section 2 text>

#CHAPTER Chapter 2 Title
#SECTION Section 1 Title
<section text>
```

Run the tagger
--------------

```bash
cd tagging
python tag_book.py
```

Common options:

```bash
# Process only the next 2 chapters
python tag_book.py --chapters 2

# Increase max tool-call rounds per section (default 10)
python tag_book.py --max-tool-rounds 12

# Start fresh from chapter 0, ignoring existing checkpoint
python tag_book.py --reset

# Custom checkpoint file/dir
python tag_book.py --checkpoint checkpoints/checkpoint_latest.json --checkpoint-dir checkpoints
```

Outputs
-------
- Checkpoints directory: `tagging/checkpoints/`
  - `checkpoint_latest.json` – resumable progress pointer
  - `checkpoint_YYYYMMDD_HHMMSS.json` – timestamped snapshots
  - `tags.json` – consolidated tags across all processed sections

Data shape (example)
--------------------
Each section entry in `checkpoints/tags.json` looks like:

```json
{
  "chapter_idx": 0,
  "chapter_title": "Chapter 1 Title",
  "section_idx": 1,
  "section_title": "Section 2 Title",
  "symptoms": [
    {"name_zh": "頭痛", "name_en": "headache"}
  ],
  "formulas": [
    {"name_zh": "桂枝湯", "name_en": "Cinnamon Twig Decoction"}
  ]
}
```

Troubleshooting
---------------
- Missing API key: set `OPENAI_API_KEY` (env or `.env`).
- Book file not found: ensure `tagging/data/book.txt` exists.
- Rate limits/temporary errors: the script retries with backoff automatically.
- Model selection: set `OPENAI_MODEL` if you need a different model; default is `gpt-4.1`.

Notes
-----
- The script processes sections one by one, saving after each chapter. You can safely stop and re-run to resume.
- Categories are recorded only when explicitly mentioned in the text, with both Chinese (`name_zh`) and English (`name_en`) names.