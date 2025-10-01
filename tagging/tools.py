symptom_tags_params = {
    "type": "array",
    "description": "List of symptoms (症狀) mentioned in the section",
    "items": {
        "type": "object",
        "properties": {
            "name_zh": {
                "type": "string",
                "description": "Chinese name of the symptom (症狀)",
            },
            "name_en": {
                "type": "string",
                "description": "English name of the symptom (症狀)",
            },
        },
        "required": ["name_zh", "name_en"],
    },
}

herb_tags_params = {
    "type": "array",
    "description": "List of herbs (中藥) mentioned in the section",
    "items": {
        "type": "object",
        "properties": {
            "name_zh": {
                "type": "string",
                "description": "Chinese name of the herb (中藥)",
            },
            "name_en": {
                "type": "string",
                "description": "English name of the herb (中藥)",
            },
        },
        "required": ["name_zh", "name_en"],
    },
}

formula_tags_params = {
    "type": "array",
    "description": "List of formulas (方劑) mentioned in the section",
    "items": {
        "type": "object",
        "properties": {
            "name_zh": {
                "type": "string",
                "description": "Chinese name of the formula (方劑)",
            },
            "name_en": {
                "type": "string",
                "description": "English name of the formula (方劑)",
            },
        },
        "required": ["name_zh", "name_en"],
    },
}

pulse_tags_params = {
    "type": "array",
    "description": "List of pulse diagnoses (脈診) mentioned in the section",
    "items": {
        "type": "object",
        "properties": {
            "name_zh": {
                "type": "string",
                "description": "Chinese name of the pulse diagnosis (脈診)",
            },
            "name_en": {
                "type": "string",
                "description": "English name of the pulse diagnosis (脈診)",
            },
        },
        "required": ["name_zh", "name_en"],
    },
}

tongue_tags_params = {
    "type": "array",
    "description": "List of tongue diagnoses (舌診) mentioned in the section",
    "items": {
        "type": "object",
        "properties": {
            "name_zh": {
                "type": "string",
                "description": "Chinese name of the tongue diagnosis (舌診)",
            },
            "name_en": {
                "type": "string",
                "description": "English name of the tongue diagnosis (舌診)",
            },
        },
        "required": ["name_zh", "name_en"],
    },
}

syndrome_tags_params = {
    "type": "array",
    "description": "List of syndromes (證候) mentioned in the section",
    "items": {
        "type": "object",
        "properties": {
            "name_zh": {
                "type": "string",
                "description": "Chinese name of the syndrome (證候)",
            },
            "name_en": {
                "type": "string",
                "description": "English name of the syndrome (證候)",
            },
        },
        "required": ["name_zh", "name_en"],
    },
}

pathogen_tags_params = {
    "type": "array",
    "description": "List of pathogens (病因) mentioned in the section",
    "items": {
        "type": "object",
        "properties": {
            "name_zh": {
                "type": "string",
                "description": "Chinese name of the pathogen (病因)",
            },
            "name_en": {
                "type": "string",
                "description": "English name of the pathogen (病因)",
            },
        },
        "required": ["name_zh", "name_en"],
    },
}

treatment_tags_params = {
    "type": "array",
    "description": "List of treatment strategies (治法) mentioned in the section",
    "items": {
        "type": "object",
        "properties": {
            "name_zh": {
                "type": "string",
                "description": "Chinese name of the treatment strategy (治法)",
            },
            "name_en": {
                "type": "string",
                "description": "English name of the treatment strategy (治法)",
            },
        },
        "required": ["name_zh", "name_en"],
    },
}

meridian_tags_params = {
    "type": "array",
    "description": "List of meridians (經絡) mentioned in the section",
    "items": {
        "type": "object",
        "properties": {
            "name_zh": {
                "type": "string",
                "description": "Chinese name of the meridian (經絡)",
            },
            "name_en": {
                "type": "string",
                "description": "English name of the meridian (經絡)",
            },
        },
        "required": ["name_zh", "name_en"],
    },
}

organ_tags_params = {
    "type": "array",
    "description": "List of organs (臟腑) mentioned in the section",
    "items": {
        "type": "object",
        "properties": {
            "name_zh": {
                "type": "string",
                "description": "Chinese name of the organ (臟腑)",
            },
            "name_en": {
                "type": "string",
                "description": "English name of the organ (臟腑)",
            },
        },
        "required": ["name_zh", "name_en"],
    },
}

acupoint_tags_params = {
    "type": "array",
    "description": "List of acupoints (穴位) mentioned in the section",
    "items": {
        "type": "object",
        "properties": {
            "name_zh": {
                "type": "string",
                "description": "Chinese name of the acupoint (穴位)",
            },
            "name_en": {
                "type": "string",
                "description": "English name of the acupoint (穴位)",
            },
        },
        "required": ["name_zh", "name_en"],
    },
}

element_tags_params = {
    "type": "array",
    "description": "List of elements (五行) mentioned in the section",
    "items": {
        "type": "object",
        "properties": {
            "name_zh": {
                "type": "string",
                "description": "Chinese name of the element (五行)",
            },
            "name_en": {
                "type": "string",
                "description": "English name of the element (五行)",
            },
        },
        "required": ["name_zh", "name_en"],
    },
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tag_section",
            "description": "Tag a section of text by various traditional Chinese medicine categories. \
                            For each category, tag every occurrence of the category in the section text. \
                            If no tags for a category are found, return an empty list for that category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symptom_tags": symptom_tags_params,
                    "herb_tags": herb_tags_params,
                    "formula_tags": formula_tags_params,
                    "pulse_tags": pulse_tags_params,
                    "tongue_tags": tongue_tags_params,
                    "syndrome_tags": syndrome_tags_params,
                    "pathogen_tags": pathogen_tags_params,
                    "treatment_tags": treatment_tags_params,
                    "meridian_tags": meridian_tags_params,
                    "organ_tags": organ_tags_params,
                    "acupoint_tags": acupoint_tags_params,
                    "element_tags": element_tags_params,
                },
                "required": [
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
                ],
            },
        },
    }
]
