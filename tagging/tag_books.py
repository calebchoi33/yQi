import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

from endpoint import chat_with_retry
from parse_books import parse_book
from tooling import build_tools_schema, make_tool_dispatch


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(
        description="Tag TCM books sections by symptoms, syndromes, etc."
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=10,
        help="Max tool-call rounds per section",
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
        default="output",
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
        "You are analyzing a section of a Traditional Chinese Medicine book to identify various TCM categories within that section. "
        "Your task is to tag this section with relevant items from the following 12 categories:\n\n"
        "CATEGORIES TO IDENTIFY:\n"
        "• symptom 症狀 - Objective symptoms and subjective experience described by patients (e.g. 頭痛、發燒、面色蒼白)\n"
        "• herb 中藥 - Single medicinal substances, including plant, mineral, or animal material (e.g. 桂枝、白芍)\n"
        "• formula 方劑 - Herbal prescriptions made up of multiple herbs (e.g. 桂枝湯、小青龍湯)\n"
        "• pulse 脈診 - Pulse qualities (e.g. 浮、沉、遲、數、弦、滑)\n"
        "• tongue 舌診 - Tongue body and coating descriptions (e.g. 舌頭胖大、舌淡、苔黃膩)\n"
        "• syndrome 證候 - Patterns of disharmony (e.g. 太陽證、陽明證、太陽中風、太陽溫病)\n"
        "• pathogen 病因 - Causes of disease (e.g. 風寒、氣滯、血瘀、陰虛、外傷)\n"
        "• treatment 治法 - Treatment strategies (e.g. 攻下、發汗、和解少陽、溫補脾腎)\n"
        "• meridian 經絡 - Regular and special meridians (e.g. 手太陰肺經, 督脈)\n"
        "• organ 臟腑 - Organs (e.g. 心、小腸、胃、腎)\n"
        "• acupoint 穴位 - Acupuncture points (e.g. 合谷、足三里)\n"
        "• element 五行 - Five elements (e.g. 木、火、土、金、水)\n\n"
        "INSTRUCTIONS:\n"
        "- Read the section carefully and identify all items from these categories\n"
        "- For each item, provide the original Chinese text (name_zh) and an English translation (name_en)\n"
        "- Use commonly understood English medical terms\n"
        "- Be specific but concise - avoid overly broad or vague terms\n"
        "- If the section mentions multiple items from the same category, tag all of them\n"
        "- If no items from a category are mentioned, omit that category from your response\n"
        "- Only include items that are explicitly mentioned in the text\n\n"
        "EXAMPLES:\n"
        "Text: '患者出現頭痛、發燒，脈象浮數，舌苔黃膩，診為太陽中風證，用桂枝湯治療。'\n"
        "Tags:\n"
        "- symptoms: [{'name_zh': '頭痛', 'name_en': 'headache'}, {'name_zh': '發燒', 'name_en': 'fever'}]\n"
        "- pulses: [{'name_zh': '浮數', 'name_en': 'floating and rapid pulse'}]\n"
        "- tongues: [{'name_zh': '舌苔黃膩', 'name_en': 'yellow greasy tongue coating'}]\n"
        "- syndromes: [{'name_zh': '太陽中風證', 'name_en': 'Taiyang wind syndrome'}]\n"
        "- formulas: [{'name_zh': '桂枝湯', 'name_en': 'Cinnamon Twig Decoction'}]\n\n"
        "For each category, use the appropriate tag_section_<category> tools to record your findings for this section."
    )

    def __init__(self, book_path: str, output_dir: str) -> None:
        self.book_path = book_path
        book_name = os.path.splitext(os.path.basename(book_path))[0]
        self.checkpoint_path = os.path.join(
            "checkpoints", f"{book_name}_checkpoint_latest.json"
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
        category: str,
        tags: List[Dict[str, str]],
    ) -> None:
        """Add tags to a section in the tagging system.

        Args:
            chapter_idx: The index of the chapter.
            chapter_title: The title of the chapter.
            section_idx: The index of the section.
            section_title: The title of the section.
            category: The category of the tags.
            tags: The list of tags for this category.
        """

        # Create the section if it doesn't exist
        if (chapter_idx, section_idx) not in self.tags_dict:
            self.tags_dict[(chapter_idx, section_idx)] = {
                "chapter_idx": chapter_idx,
                "chapter_title": chapter_title,
                "section_idx": section_idx,
                "section_title": section_title,
            }

        # Update the section tags
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
        tags = checkpoint.get("tags", {}).get("sections", [])
        self.tags_dict = {(tag["chapter_idx"], tag["section_idx"]): tag for tag in tags}

        # Restore the duration
        self.duration_secs = checkpoint.get("duration_secs", 0.0)

        next_chapter_idx = checkpoint.get("next_chapter_idx", 0)
        print(f"Loaded checkpoint with {len(self.tags_dict)} tagged sections")
        print(f"Next chapter index: {next_chapter_idx + 1}")

        return next_chapter_idx

    def save_checkpoint(self, next_chapter_idx: int) -> None:
        """Save the tags checkpoint file."""

        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            checkpoint = {
                "next_chapter_idx": next_chapter_idx,
                "tags": self.to_tags_list(),
                "updated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "duration_secs": self.duration_secs,
            }
            json.dump(checkpoint, f, ensure_ascii=False, indent=4)

    def save_tags(self) -> None:
        """Save the tags to tags.json file."""

        tags = self.to_tags_list()
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(tags, f, ensure_ascii=False, indent=4)

    def tag_book(self, reset: bool, max_tool_rounds: int) -> None:
        """Tag the book."""

        # Initialize progress from checkpoint
        next_chapter_idx = 0
        if not reset:
            next_chapter_idx = self.load_checkpoint()

        # Parse the book if it doesn't exist
        if not os.path.exists(self.parsed_book_path):
            parse_book(self.book_path)

        # Load the parsed book
        with open(self.parsed_book_path, "r", encoding="utf-8") as f:
            parsed_book = json.load(f)

        total_chapters = len(parsed_book)

        # Build tools schema and dispatch
        tools_schema = build_tools_schema()
        dispatch = make_tool_dispatch(self)

        # Iterate chapters
        for ch_idx in range(next_chapter_idx, total_chapters):
            chapter_start_time = time.time()

            print(f"Processing chapter {ch_idx + 1} out of {total_chapters}")
            chapter = parsed_book[ch_idx]
            sects = chapter["sections"]

            # Iterate sections
            for s_idx, section in enumerate(sects):
                print(f"Processing section {s_idx + 1} out of {len(sects)}")

                # Create the initial message for the section
                msgs: List[Dict[str, Any]] = [
                    {"role": "system", "content": self.SYSTEM_INSTRUCTIONS},
                    {
                        "role": "user",
                        "content": (
                            f"Please analyze the following section and identify any and all traditional Chinese medicine categories mentioned.\n"
                            f"When you find items, call the appropriate tool among: \n"
                            f"- tag_section_symptoms, tag_section_herbs, tag_section_formulas, tag_section_pulses, tag_section_tongues,\n"
                            f"  tag_section_syndromes, tag_section_pathogens, tag_section_treatments, tag_section_meridians,\n"
                            f"  tag_section_organs, tag_section_acupoints, tag_section_elements.\n"
                            f"Provide both Chinese and English names for each item.\n\n"
                            f"TEXT:{section['section_title']}\n{section['section_text']}\n"
                        ),
                    },
                ]

                # Process the section with the LLM tool calls
                rounds = 0
                while rounds < max_tool_rounds:
                    resp = chat_with_retry(msgs, tools=tools_schema, tool_choice="auto")
                    assistant_msg = resp.choices[0].message
                    msgs.append(
                        {
                            "role": "assistant",
                            "content": assistant_msg.content or "",
                            "tool_calls": [
                                tc.model_dump()
                                for tc in (assistant_msg.tool_calls or [])
                            ],
                        }
                    )

                    # Break if no tool calls
                    if not assistant_msg.tool_calls:
                        break

                    # Execute tool calls immediately after assistant tool_calls
                    for tc in assistant_msg.tool_calls or []:

                        # Get the tool name and arguments
                        name = tc.function.name
                        args = json.loads(tc.function.arguments)

                        # Execute the tool
                        tool_function = dispatch.get(name)
                        print(f"Calling tool {name}: {args}")

                        # Add the chapter_idx, chapter_title, section_idx, section_title to the args
                        args["chapter_idx"] = ch_idx
                        args["chapter_title"] = chapter["chapter_title"]
                        args["section_title"] = section["section_title"]
                        args["section_idx"] = section["section_idx"]

                        # Execute the tool
                        result = tool_function(args)

                        # Add the tool result to the messages
                        msgs.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": name,
                                "content": str(result),
                            }
                        )

                    rounds += 1

                # Pause briefly between sections to reduce request pressure
                time.sleep(1)

            # Update the duration
            self.duration_secs += time.time() - chapter_start_time
            # Save the checkpoint after finishing this chapter
            self.save_checkpoint(ch_idx + 1)

        # Final save at end of run
        self.save_tags()


def orchestrate_tagging(
    book_paths: List[str], reset: bool, max_tool_rounds: int, output_dir: str
) -> None:
    """Orchestrate the tagging of the books."""
    for book_path in book_paths:
        tagging_system = TCMBookTaggingSystem(book_path, output_dir)
        tagging_system.tag_book(reset, max_tool_rounds)

        # Pause briefly between books to reduce request pressure
        time.sleep(10)


if __name__ == "__main__":
    load_dotenv()

    print("Starting TCM Book Tagging System...")
    args = parse_arguments()
    orchestrate_tagging(
        args.book_paths, args.reset, args.max_tool_rounds, args.output_dir
    )
    print("TCM Book Tagging System completed.")
