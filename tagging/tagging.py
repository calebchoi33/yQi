import os
import json
from typing import Any, Dict, List, Tuple
from datetime import datetime


# TODO: Add mutex for each section to avoid race conditions
class TaggingSystem:
    """
    A system for tagging sections of a book with symptoms.

    data: Dict[Tuple[int, int], Dict[str, Any]] = {} # (chapter_idx, section_idx) -> tagged_section
    Example:
    ```
    {
        (1, 1): {
            "chapter_idx": 1,
            "chapter_title": "Chapter 1",
            "section_idx": 1,
            "section_title": "Section 1",
            "symptoms": [
                {
                    "name_zh": "症狀1",
                    "name_en": "Symptom 1",
                },
                {
                    "name_zh": "症狀2",
                    "name_en": "Symptom 2",
                },
            ]
        },
        ...
    }
    ```
    """

    def __init__(self) -> None:
        self.data: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def create_or_update_tagged_section(
        self,
        chapter_idx: int,
        chapter_title: str,
        section_idx: int,
        section_title: str,
        category: str,
        tags: List[Dict[str, str]],
    ) -> None:
        """Add or update a tagged section in the tagging system.

        Args:
            chapter_idx: The index of the chapter.
            chapter_title: The title of the chapter.
            section_idx: The index of the section.
            section_title: The title of the section.
            category: The category of the tags.
            tags: The list of tags for this category.
        """

        # Create the section if it doesn't exist
        if (chapter_idx, section_idx) not in self.data:
            self.data[(chapter_idx, section_idx)] = {
                "chapter_idx": chapter_idx,
                "chapter_title": chapter_title,
                "section_idx": section_idx,
                "section_title": section_title,
            }

        self.data[(chapter_idx, section_idx)][category] = tags

    def to_json(self) -> Dict[str, Any]:
        """Convert the tagging system to a JSON list of tagged sections."""
        return {"sections": list(self.data.values())}


def save_tags_and_checkpoint(
    tagging_system: TaggingSystem, next_chapter_idx: int, checkpoint_dir: str
) -> str:
    """
    Persist tags.json and both latest and timestamped checkpoints.

    Args:
        tagging_system: The tagging system to save.
        next_chapter_idx: The index of the next chapter to process.
        checkpoint_dir: The directory to save the checkpoint.
    """
    # Create the checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the tagging system to tags.json
    summary = tagging_system.to_json()
    with open(os.path.join(checkpoint_dir, "tags.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Create the latest checkpoint payload
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_payload = {
        "next_chapter_idx": next_chapter_idx,
        "tags": summary,
        "updated_at": ts,
    }

    # Save to the latest checkpoint
    with open(
        os.path.join(checkpoint_dir, "checkpoint_latest.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(latest_payload, f, ensure_ascii=False, indent=2)

    # Save to the timestamped checkpoint
    ts_path = os.path.join(checkpoint_dir, f"checkpoint_{ts}.json")
    with open(ts_path, "w", encoding="utf-8") as f:
        json.dump(latest_payload, f, ensure_ascii=False, indent=2)

    return ts_path
