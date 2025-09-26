import json
from typing import Any, Dict, List


def build_tools_schema() -> List[Dict[str, Any]]:
    """Build per-category tool schemas for the tagging system."""

    def _tags_params() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "description": "List of items for this category",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name_zh": {
                                "type": "string",
                                "description": "Chinese name",
                            },
                            "name_en": {
                                "type": "string",
                                "description": "English name",
                            },
                        },
                        "required": ["name_zh", "name_en"],
                    },
                },
            },
            "required": ["tags"],
        }

    def _func(name: str, description: str) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": _tags_params(),
            },
        }

    return [
        _func("tag_section_symptoms", "Tag a section with symptoms (症狀)."),
        _func("tag_section_herbs", "Tag a section with herbs (中藥)."),
        _func("tag_section_formulas", "Tag a section with formulas (方劑)."),
        _func("tag_section_pulses", "Tag a section with pulse qualities (脈診)."),
        _func("tag_section_tongues", "Tag a section with tongue descriptions (舌診)."),
        _func("tag_section_syndromes", "Tag a section with syndromes/patterns (證候)."),
        _func(
            "tag_section_pathogens", "Tag a section with pathogens/etiologies (病因)."
        ),
        _func(
            "tag_section_treatments", "Tag a section with treatment strategies (治法)."
        ),
        _func("tag_section_meridians", "Tag a section with meridians (經絡)."),
        _func("tag_section_organs", "Tag a section with organs (臟腑)."),
        _func("tag_section_acupoints", "Tag a section with acupoints (穴位)."),
        _func("tag_section_elements", "Tag a section with five elements (五行)."),
    ]


def make_tool_dispatch(tag_writer: Any):
    """Make the tool dispatch for the tagging system."""

    def _validate_args(args: Dict[str, Any]) -> None:
        required_args = [
            "chapter_idx",
            "chapter_title",
            "section_idx",
            "section_title",
            "tags",
        ]
        for arg in required_args:
            if arg not in args:
                raise ValueError(f"{arg} is required")

    def _handler(category: str):
        def fn(args: Dict[str, Any]) -> str:
            _validate_args(args)

            tag_writer.add_tags(
                args["chapter_idx"],
                args["chapter_title"],
                args["section_idx"],
                args["section_title"],
                category,
                args["tags"],
            )
            return json.dumps(
                {"status": "ok", "category": category, "tags_count": len(args["tags"])},
                ensure_ascii=False,
            )

        return fn

    return {
        "tag_section_symptoms": _handler("symptoms"),
        "tag_section_herbs": _handler("herbs"),
        "tag_section_formulas": _handler("formulas"),
        "tag_section_pulses": _handler("pulses"),
        "tag_section_tongues": _handler("tongues"),
        "tag_section_syndromes": _handler("syndromes"),
        "tag_section_pathogens": _handler("pathogens"),
        "tag_section_treatments": _handler("treatments"),
        "tag_section_meridians": _handler("meridians"),
        "tag_section_organs": _handler("organs"),
        "tag_section_acupoints": _handler("acupoints"),
        "tag_section_elements": _handler("elements"),
    }
