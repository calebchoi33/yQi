import argparse
import json
import os
import re
from typing import Any, Dict, List


def parse_book(path: str) -> List[Dict[str, Any]]:
    """
    Parses the book into chapters and sections.
    Save the parsed book to a JSON file in the output directory.

    Args:
        path: Path to the book file.
    """

    # Check if the book file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"book file not found at {path}")

    print(f"Parsing book {path}...")

    # Read the book file
    with open(path, "r", encoding="utf-8") as f:
        file_content = f.read()

    # PARSE THE BOOK BY CHAPTERS AND SECTIONS
    parsed_book = []
    ch_blocks = re.split(r"^#CHAPTER\s*", file_content, flags=re.MULTILINE)
    for ch_idx, ch_block in enumerate(ch_blocks):
        if not ch_block:
            continue

        # Separate chapter title from text
        lines = ch_block.splitlines()
        ch_title = lines[0].strip()
        ch_text = "\n".join(lines[1:])

        # Split sections within the chapter
        sections = []
        sec_blocks = re.split(r"^#SECTION\s*", ch_text, flags=re.MULTILINE)

        # Case 1: No #SECTION in chapter. Treat entire chapter text as a single, untitled section
        if len(sec_blocks) == 1:
            chapter_description = sec_blocks[0].strip()
            if not chapter_description:
                chapter_description = "<No text in section>"

            sections.append(
                {
                    "section_idx": 0,
                    "section_title": "<Chapter description>",
                    "section_text": chapter_description,
                }
            )
        else:
            # Case 2: There are sections. The first block may be the chapter description
            running_idx = 0
            chapter_description = sec_blocks[0].strip()
            if chapter_description:
                sections.append(
                    {
                        "section_idx": running_idx,
                        "section_title": "<Chapter description>",
                        "section_text": chapter_description,
                    }
                )
                running_idx += 1

            # Parse each real section: first non-empty line is the title, rest is content
            for sec_block in sec_blocks[1:]:
                block = sec_block.strip()
                if not block:
                    continue

                sec_lines = sec_block.splitlines()
                sec_title = sec_lines[0].strip()
                sec_text = "\n".join(sec_lines[1:]).strip()
                if not sec_text:
                    sec_text = "<No text in section>"

                sections.append(
                    {
                        "section_idx": running_idx,
                        "section_title": sec_title,
                        "section_text": sec_text,
                    }
                )
                running_idx += 1

        # Add the chapter to the parsed book
        parsed_book.append(
            {
                "chapter_idx": ch_idx,
                "chapter_title": ch_title,
                "sections": sections,
            }
        )

    return parsed_book


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse a book(s) into chapters and sections and save the result to a json."
    )
    parser.add_argument(
        "--book-paths",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to the book file(s)",
    )
    args = parser.parse_args()

    for book_path in args.book_paths:
        parsed_book = parse_book(book_path)

        # Save the parsed book to a JSON file
        book_dir = os.path.dirname(book_path)
        book_name = os.path.splitext(os.path.basename(book_path))[0]
        parsed_book_path = os.path.join(book_dir, f"{book_name}_parsed.json")
        with open(parsed_book_path, "w", encoding="utf-8") as f:
            json.dump(parsed_book, f, ensure_ascii=False, indent=4)
