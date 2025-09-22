import json
import os
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from utils import chat_with_retry, parse_book
from tagging import TaggingSystem, save_tags_and_checkpoint
from tooling import build_tools_schema, make_tool_dispatch


load_dotenv()

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
    "Use the tag_section tool to record your findings for this section."
)


def run_tagging_system(
    *,
    max_tool_rounds: int = 8,
    chapters_to_process: Optional[int] = None,
    checkpoint_path: str = "checkpoints/checkpoint_latest.json",
    checkpoint_dir: str = "checkpoints",
    reset: bool = False,
    book_path: str = "data/book.txt",
) -> None:
    """Run the tagging system."""

    # temp:
    reset = True

    # Initialize tagging system and progress
    tagging_system = TaggingSystem()
    next_chapter_idx = 0

    # Load checkpoint if present
    if not reset and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)

            # Restore tagging system
            data = checkpoint.get("tags", {})
            if "sections" in data:
                tagging_system.tagged_sections = data["sections"]
            next_chapter_idx = int(checkpoint.get("next_chapter_idx", 0))
        except Exception:
            next_chapter_idx = 0

    # Read in book data
    parsed_book = parse_book(book_path)
    total_chapters = len(parsed_book)

    # Build tools schema and dispatch
    tools_schema = build_tools_schema()
    dispatch = make_tool_dispatch(tagging_system)

    # Determine number of chapters to process this run
    remaining = total_chapters - next_chapter_idx
    to_process = (
        remaining
        if (chapters_to_process is None or chapters_to_process <= 0)
        else min(remaining, chapters_to_process)
    )

    # Iterate chapters and process each section individually
    for ch_idx in range(next_chapter_idx, next_chapter_idx + to_process):
        chapter = parsed_book[ch_idx]
        sects = chapter["sections"]

        for s_idx, section in enumerate(sects):
            # Print the section to be processed
            print(
                f"[processing] chapter {ch_idx + 1} out of {total_chapters}, section {s_idx + 1} out of {len(sects)}"
            )

            # Create the initial message for the section
            msgs: List[Dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {
                    "role": "user",
                    "content": (
                        f"Please analyze the following section and identify any and all TCM categories mentioned.\n"
                        f"When you find items, call the appropriate tool among: \n"
                        f"- tag_section_symptoms, tag_section_herbs, tag_section_formulas, tag_section_pulses, tag_section_tongues,\n"
                        f"  tag_section_syndromes, tag_section_pathogens, tag_section_treatments, tag_section_meridians,\n"
                        f"  tag_section_organs, tag_section_acupoints, tag_section_elements.\n\n"
                        f"TEXT:\n{section['section_text']}\n\n"
                        f"Provide both Chinese and English names for each item.\n"
                    ),
                },
            ]

            # Process the section
            rounds = 0
            while rounds < max_tool_rounds:
                resp = chat_with_retry(msgs, tools=tools_schema, tool_choice="auto")
                assistant_msg = resp.choices[0].message
                msgs.append(
                    {
                        "role": "assistant",
                        "content": assistant_msg.content or "",
                        "tool_calls": [
                            tc.model_dump() for tc in (assistant_msg.tool_calls or [])
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
                    raw_args = tc.function.arguments or "{}"
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {}

                    # Execute the tool
                    tool_function = dispatch.get(name)
                    if not tool_function:
                        result = f"WARNING: Tool not implemented: {name}"
                    else:
                        # Print the tool call for logging
                        print(f"[call] {name}: {args}")

                        # Add the chapter_idx, chapter_title, section_idx, section_title to the args
                        args["chapter_idx"] = ch_idx
                        args["chapter_title"] = chapter["chapter_title"]
                        args["section_title"] = section["section_title"]
                        args["section_idx"] = section["section_idx"]

                        # Execute the tool
                        try:
                            result = tool_function(args)
                        except Exception as e:  # safety against tool crashes
                            result = f"ERROR: Tool {name} failed: {e}"
                            raise e

                    # Add the tool result to the messages
                    msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": str(result),
                        }
                    )

                    # Print brief warning/error feedback
                    if isinstance(result, str):
                        if result.startswith("WARNING"):
                            print(f"[warn] {result}")
                        elif result.startswith("ERROR"):
                            print(f"[error] {result}")

                rounds += 1

            # Pause briefly between sections to reduce request pressure
            time.sleep(1)

        # Autosave after finishing this chapter
        next_chapter_idx = ch_idx + 1
        checkpoint_path = save_tags_and_checkpoint(
            tagging_system=tagging_system,
            next_chapter_idx=next_chapter_idx,
            checkpoint_dir=checkpoint_dir,
        )
        print(
            f"[save] chapter {next_chapter_idx}/{total_chapters} checkpoint -> {checkpoint_path}"
        )

    # Update progress
    next_chapter_idx += to_process

    # Final save at end of run
    checkpoint_path = save_tags_and_checkpoint(
        tagging_system=tagging_system,
        next_chapter_idx=next_chapter_idx,
        checkpoint_dir=checkpoint_dir,
    )

    # Print concise counts
    print(
        json.dumps(
            {
                "sections_tagged": len(tagging_system.tagged_sections),
                "next_chapter_idx": next_chapter_idx,
                "chapters_total": total_chapters,
                "checkpoint_latest": checkpoint_path,
                "tags_saved": "tags.json",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tag TCM book sections with symptoms from book.txt"
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=10,
        help="Max tool-call rounds per section",
    )
    parser.add_argument(
        "--chapters",
        type=int,
        default=0,
        help="Number of chapters to process this run (0 = all remaining)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/checkpoint_latest.json",
        help="Path to latest checkpoint file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to store timestamped checkpoints",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ignore existing checkpoint and start from chapter 0",
    )

    args = parser.parse_args()

    run_tagging_system(
        max_tool_rounds=args.max_tool_rounds,
        chapters_to_process=args.chapters,
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        reset=args.reset,
    )
