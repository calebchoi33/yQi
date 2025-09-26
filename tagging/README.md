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

Prepare the book
----------------
Format your traditional Chinese medicine source text with chapter and section markers:

```text
#CHAPTER
Chapter 1 Title
<chapter 1 description>

#SECTION
Section 1 Title
<section 1 text>

#SECTION
Section 2 Title
<section 2 text>

#CHAPTER
Chapter 2 Title

#SECTION
Section 1 Title
<section text>
```

Run the tagger
--------------

```bash
$ python tagging/tag_book.py --book-paths books/tcm_book1.txt books/tcm_book2.txt
```

Outputs
-------
- Checkpoints: `tagging/checkpoints/<book_name>_checkpoint_latest.json`
- Final tags: `tagging/output/<book_name>_tags.json`
- Parsed book JSON: `<same directory as the book>/<book_name>_parsed.json`

Flatten per-book unique tag lists
---------------------------------
After tagging, create a per-book unique list of items per category:

```bash
$ python tagging/flatten_tags.py --output-dir tagging/output
```

This writes `tagging/output/<book_name>_tags_categories_list.json` containing de-duplicated items per category.

Parse books
---------------------
If you only want to build the parsed JSON for one or more books without tagging:

```bash
$ python tagging/parse_books.py --book-paths books/tcm_book1.txt books/tcm_book2.txt
```

This will create the parsed book JSON: `<same directory as the book>/<book_name>_parsed.json`