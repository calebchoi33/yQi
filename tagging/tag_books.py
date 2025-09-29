import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

from endpoint import chat_with_retry
from parse_books import parse_book
from tools import TOOLS


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(
        description="Tag TCM books sections by symptoms, syndromes, etc."
    )
    parser.add_argument(
        "--book-paths",
        type=str,
        required=True,
        nargs="+",
        help="Path(s) to the book txt files(s)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Ignore existing checkpoints and restart tagging",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tagging/output",
        help="Output directory for the tagged books",
    )
    return parser.parse_args()


class TCMBookTaggingSystem:
    """
    A system for tagging sections of a TCM book with symptoms, syndromes, etc.

    Args:
        book_path: str
            the path to the book file.

        output_dir: str
            the path to the output directory for this book.

    Attributes:
        book_path: str
            the path to the book file.

        checkpoint_path: str
            the path to the checkpoint file for this book.

        output_path: str
            the path to the output file for this book.

        parsed_book_path: str
            the path to the parsed book file.

        tags: Dict[Tuple[int, int], Dict[str, Any]] = {}
            the section tags themselves, stored as a dictionary of (chapter_idx, section_idx) -> tagged_section for easy lookup.

    """

    SYSTEM_INSTRUCTIONS = (
        "You are analyzing a section of a Traditional Chinese Medicine book to identify occurences of various TCM categories within the section. "
        "Your task is to tag the section, which means identifying the occurences of the following 12 categories in this section:\n\n"
        "CATEGORIES TO IDENTIFY:\n"
        "• symptoms 症狀 - Objective symptoms and subjective experience described by patients (e.g. 頭痛、發燒、面色蒼白)\n"
        "• herbs 中藥 - Single medicinal substances, including plant, mineral, or animal material (e.g. 桂枝、白芍)\n"
        "• formulas 方劑 - Herbal prescriptions made up of multiple herbs (e.g. 桂枝湯、小青龍湯)\n"
        "• pulses 脈診 - Pulse qualities (e.g. 浮、沉、遲、數、弦、滑)\n"
        "• tongues 舌診 - Tongue body and coating descriptions (e.g. 舌頭胖大、舌淡、苔黃膩)\n"
        "• syndromes 證候 - Patterns of disharmony (e.g. 太陽證、陽明證、太陽中風、太陽溫病)\n"
        "• pathogens 病因 - Causes of disease (e.g. 風寒、氣滯、血瘀、陰虛、外傷)\n"
        "• treatments 治法 - Treatment strategies (e.g. 攻下、發汗、和解少陽、溫補脾腎)\n"
        "• meridians 經絡 - Regular and special meridians (e.g. 手太陰肺經, 督脈)\n"
        "• organs 臟腑 - Organs (e.g. 心、小腸、胃、腎)\n"
        "• acupoints 穴位 - Acupuncture points (e.g. 合谷、足三里)\n"
        "• elements 五行 - Five elements (e.g. 木、火、土、金、水)\n\n"
        "INSTRUCTIONS:\n"
        "- Read the section carefully and identify all occurences of the above categories\n"
        "- For each occurence, provide the original Chinese text (name_zh) and an English translation (name_en)\n"
        "- Use commonly understood English medical terms\n"
        "- Be specific but concise - avoid overly broad or vague terms\n"
        "- If the section mentions multiple occurences of the same category, tag all of them\n"
        "- If no occurences of a category are mentioned, return an empty list for that category\n"
        "- Only include occurences that are explicitly mentioned in the text\n\n"
        "EXAMPLES:\n"
        "Text: '患者出現頭痛、發燒，脈象浮數，舌苔黃膩，診為太陽中風證，用桂枝湯治療。'\n"
        "Tags:\n"
        "- symptoms: [{'name_zh': '頭痛', 'name_en': 'headache'}, {'name_zh': '發燒', 'name_en': 'fever'}]\n"
        "- pulses: [{'name_zh': '浮數', 'name_en': 'floating and rapid pulse'}]\n"
        "- tongues: [{'name_zh': '舌苔黃膩', 'name_en': 'yellow greasy tongue coating'}]\n"
        "- syndromes: [{'name_zh': '太陽中風證', 'name_en': 'Taiyang wind syndrome'}]\n"
        "- formulas: [{'name_zh': '桂枝湯', 'name_en': 'Cinnamon Twig Decoction'}]\n\n"
    )

    def __init__(self, book_path: str, output_dir: str) -> None:
        self.book_path = book_path
        book_name = os.path.splitext(os.path.basename(book_path))[0]
        self.checkpoint_path = os.path.join(
            "tagging/checkpoints", f"{book_name}_checkpoint_latest.json"
        )
        self.output_path = os.path.join(output_dir, f"{book_name}_tags.json")
        self.parsed_book_path = os.path.join(
            os.path.dirname(self.book_path), f"{book_name}_parsed.json"
        )

        self.duration_secs = 0.0

        self.tags_dict: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def add_tags(
        self,
        chapter_idx: int,
        chapter_title: str,
        section_idx: int,
        section_title: str,
        args: Dict[str, Any],
    ) -> None:
        """Add tags to a section in the tagging system.

        Args:
            chapter_idx: The index of the chapter.
            chapter_title: The title of the chapter.
            section_idx: The index of the section.
            section_title: The title of the section.
            args: Dict containing tags by category.
        """

        # Ensure all the categories are present
        required_args = [
            "symptom_tags",
            "herb_tags",
            "formula_tags",
            "pulse_tags",
            "tongue_tags",
            "syndrome_tags",
            "pathogen_tags",
            "treatment_tags",
            "meridian_tags",
            "organ_tags",
            "acupoint_tags",
            "element_tags",
        ]
        for arg in required_args:
            if arg not in args:
                raise ValueError(f"{arg} is required")

        # Create the section
        self.tags_dict[(chapter_idx, section_idx)] = {
            "chapter_idx": chapter_idx,
            "chapter_title": chapter_title,
            "section_idx": section_idx,
            "section_title": section_title,
        }

        # Update the section tags
        for category, tags in args.items():
            if tags:
                self.tags_dict[(chapter_idx, section_idx)][category] = tags

    def to_tags_list(self) -> Dict[str, Any]:
        """Convert the tagging system to a JSON list of tagged sections."""
        return {"sections": list(self.tags_dict.values())}

    def load_checkpoint(self) -> int:
        """Load the checkpoint file, restore the tags dictionary, and return the next chapter index."""

        # Check if the checkpoint exists
        if not os.path.exists(self.checkpoint_path):
            return 0

        # Load the checkpoint
        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        # Restore the tags dictionary
        tags = checkpoint["tags"]["sections"]
        self.tags_dict = {(tag["chapter_idx"], tag["section_idx"]): tag for tag in tags}

        # Restore the duration
        self.duration_secs = checkpoint["duration_secs"]

        next_chapter_idx = checkpoint["next_chapter_idx"]
        print(f"Loaded checkpoint with {len(self.tags_dict)} tagged sections")
        print(f"Next chapter index: {next_chapter_idx}")

        return next_chapter_idx

    def save_checkpoint(self, next_chapter_idx: int) -> None:
        """Save the tags checkpoint file."""

        tags = self.to_tags_list()
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            checkpoint = {
                "next_chapter_idx": next_chapter_idx,
                "tags": tags,
                "updated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "duration_secs": self.duration_secs,
            }
            json.dump(checkpoint, f, ensure_ascii=False, indent=4)

    def save_tags(self) -> None:
        """Save the tags to tags.json file."""

        tags = self.to_tags_list()
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(tags, f, ensure_ascii=False, indent=4)

    def tag_book(self, reset: bool) -> None:
        """Tag the book."""

        # Initialize progress from checkpoint
        next_chapter_idx = 0
        if not reset:
            next_chapter_idx = self.load_checkpoint()

        # Parse the book
        parsed_book = parse_book(self.book_path)
        total_chapters = len(parsed_book)

        # Iterate chapters
        for ch_idx in range(next_chapter_idx, total_chapters):
            chapter_start_time = time.time()

            print(f"Processing chapter {ch_idx} out of {total_chapters}")
            chapter = parsed_book[ch_idx]
            sects = chapter["sections"]

            # Iterate sections
            for s_idx, section in enumerate(sects):
                print(f"Processing section {s_idx} out of {len(sects)}")

                # Create the initial message for the section
                input_msgs = [
                    {"role": "system", "content": self.SYSTEM_INSTRUCTIONS},
                    {
                        "role": "user",
                        "content": (
                            f"Please analyze the following section and identify any and all traditional Chinese medicine categories mentioned.\n"
                            f"Use the tag_section tool to tag the section with the following categories: symptom, herb, formula, pulse, tongue, syndrome, pathogen, treatment, meridian, organ, acupoint, element.\n"
                            "If no tags for a category are found, return an empty list for that category.\n"
                            "Only include items that are explicitly mentioned in the text.\n\n"
                            f"TEXT:{section['section_title']}\n{section['section_text']}\n"
                        ),
                    },
                ]

                # Process the section with the LLM tool calls
                resp = chat_with_retry(input_msgs, tools=TOOLS)
                assistant_msg = resp.choices[0].message

                # If no tool calls, raise an error
                if not assistant_msg.tool_calls:
                    raise ValueError("No tool calls")

                # There should be exactly one tool call
                if len(assistant_msg.tool_calls) != 1:
                    raise ValueError("Expected exactly one tool call")

                # Execute the tool
                tc = assistant_msg.tool_calls[0]
                function_args = json.loads(tc.function.arguments)
                print(f"Adding tags: {function_args}")
                self.add_tags(
                    chapter["chapter_idx"],
                    chapter["chapter_title"],
                    section["section_idx"],
                    section["section_title"],
                    function_args,
                )

                # Pause briefly between sections to reduce request pressure
                time.sleep(1)

            # Update the duration
            self.duration_secs += time.time() - chapter_start_time
            # Save the checkpoint after finishing this chapter
            self.save_checkpoint(ch_idx + 1)

        # Final save at end of run
        self.save_tags()


def orchestrate_tagging(book_paths: List[str], reset: bool, output_dir: str) -> None:
    """Orchestrate the tagging of the books."""
    for book_path in book_paths:
        tagging_system = TCMBookTaggingSystem(book_path, output_dir)
        tagging_system.tag_book(reset)

        # Pause briefly between books to reduce request pressure
        time.sleep(10)


if __name__ == "__main__":
    load_dotenv()

    print("Starting TCM Book Tagging System...")
    args = parse_arguments()
    orchestrate_tagging(args.book_paths, args.reset, args.output_dir)
    print("TCM Book Tagging System completed.")
