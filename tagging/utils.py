import time
import os
import re

from typing import Any, Dict, List, Optional

from endpoint import call_chat


# =============================
# Helper functions
# =============================


def chat_with_retry(
    messages: List[Dict[str, Any]],
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = "auto",
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = 3,
):
    """Call chat API with exponential backoff retry.

    Retries up to max_retries on exceptions (e.g., rate limits), waiting 1, 2, 4 ... seconds.
    """
    attempt = 0
    while True:
        try:
            return call_chat(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                model=model,
                temperature=temperature,
            )
        except Exception as e:
            if attempt >= max_retries:
                raise
            delay = 2**attempt
            print(
                f"[retry] {type(e).__name__}: sleeping {delay}s (attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(delay)
            attempt += 1


# =============================
# Book parsing
# =============================


def parse_book(path: str) -> List[Dict[str, Any]]:
    """Return the book as a list of chapters, and the sections within each chapter.

    Args:
        path: Path to the book file.

    Returns:
        A list of chapters, and the sections within each chapter.

    Example output:
    ```
        [
            {
                "chapter_idx": int,
                "chapter_title": str,
                "sections": [
                    {
                        "section_idx": int,
                        "section_title": str,
                        "section_text": str,
                    },
                    ...
                ]
            },
            ...
        ]
    ```
    """

    # Check if the book file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"book file not found at {path}")

    # Read the book file
    with open(path, "r", encoding="utf-8") as f:
        file_content = f.read()

    # Parse the book by chapters and sections within each chapter
    parsed_book: List[Dict[str, Any]] = []
    ch_blocks = re.split(r"^#CHAPTER\s*", file_content, flags=re.MULTILINE)
    for ch_idx, ch_block in enumerate(ch_blocks):
        ch_block = ch_block.strip()
        if not ch_block:
            continue

        # Separate chapter title from text
        lines = ch_block.splitlines()
        ch_title = lines[0].strip()
        ch_text = "\n".join(lines[1:])

        # Split sections within the chapter
        sections: List[Dict[str, Any]] = []
        sec_blocks = re.split(r"^#SECTION\s*", ch_text, flags=re.MULTILINE)
        for sec_idx, sec_block in enumerate(sec_blocks):
            sec_block = sec_block.strip()
            if not sec_block:
                continue

            # Separate section title from text
            sec_lines = sec_block.splitlines()
            sec_title = sec_lines[0].strip()
            sec_text = "\n".join(sec_lines[1:]).strip()
            if not sec_text:
                sec_text = "<No text in section>"

            # Add the section to the chapter
            sections.append(
                {
                    "section_idx": sec_idx,
                    "section_title": sec_title,
                    "section_text": sec_text,
                }
            )

        # Add the chapter to the parsed book
        parsed_book.append(
            {
                "chapter_idx": ch_idx,
                "chapter_title": ch_title,
                "sections": sections,
            }
        )

    return parsed_book


def format_sections_for_prompt(sections: List[Dict[str, Any]]) -> str:
    """Format sections for the prompt. Can handle single or multiple sections.

    Args:
        sections: List of sections to format. Each section is a dictionary with the following keys:
            - section_idx: int, the index of the section within the chapter
            - section_title: str, the title of the section
            - section_text: str, the text of the section

    Returns:
        A string with the sections formatted for the prompt.

    Example output for single section:
    ```
        ### SECTION_IDX: 1 ###

        <section 1 text>
    ```

    Example output for multiple sections:
    ```
        ### SECTION_IDX: 1 ###

        <section 1 text>

        ### SECTION_IDX: 2 ###

        <section 2 text>
    ```
    """
    parts = []
    for sec in sections:
        parts.append(f"### SECTION_IDX: {sec['section_idx']} ###")
        parts.append(f"{sec['section_text']}")
    return "\n\n".join(parts)
