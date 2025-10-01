import argparse
import os
import json
from typing import Dict, List, Tuple


CATEGORIES: List[str] = [
    "symptoms_tags",
    "herbs_tags",
    "formulas_tags",
    "pulses_tags",
    "tongues_tags",
    "syndromes_tags",
    "pathogens_tags",
    "treatments_tags",
    "meridians_tags",
    "organs_tags",
    "acupoints_tags",
    "elements_tags",
]


def _flatten_sections(sections: List[Dict]) -> Dict[str, List[Dict[str, str]]]:
    """Flatten the sections into a dictionary of categories and tags."""
    flattened: Dict[str, Dict[Tuple[str, str], Dict[str, str]]] = {
        category: {} for category in CATEGORIES
    }

    for section in sections:
        for category in CATEGORIES:
            tags = section.get(category, [])
            for tag in tags:
                # Every tag should have a name_zh
                if "name_zh" not in tag:
                    raise ValueError(f"name_zh not found in tag: {tag}")

                flattened[category][tag["name_zh"]] = tag

    # Convert inner dicts to ordered lists and drop empty categories
    result: Dict[str, List[Dict[str, str]]] = {}
    for category, items_map in flattened.items():
        if items_map:
            result[category] = list(items_map.values())
    return result


def flatten_tags(output_dir: str) -> None:
    """Flatten the tags from the output directory."""

    print(f"Flattening tags from {output_dir}...")
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    # Walk through the output directory
    for filename in os.listdir(output_dir):

        if not (filename.endswith("_tags.json")):
            continue

        print(f"Processing {filename}...")

        # Read the tags from the file
        input_path = os.path.join(output_dir, filename)
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Flatten the tags
        flattened = _flatten_sections(data.get("sections", []))

        # Write the flattened tags to a file
        book_name = filename.split("_tags.json")[0]
        output_filename = f"{book_name}_tags_categories_list.json"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(flattened, f, ensure_ascii=False, indent=4)

        print(f"Wrote {output_filename} ({len(flattened)} categories)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flatten per-section tags into per-book unique tag lists"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("tagging", "output"),
        help="Directory containing *_tags.json files and where *_all_tags.json will be written",
    )
    args = parser.parse_args()
    flatten_tags(args.output_dir)
