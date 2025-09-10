import json
import os
import re
import uuid
import time
from typing import Any, Dict, List, Optional, Tuple

from endpoint import call_chat, first_message_text


# =============================
# Helpers and Data Structures
# =============================


def bi(zh: str, en: str) -> str:
    """Concatenate bilingual text (Chinese first, then English)."""
    zh = (zh or "").strip()
    en = (en or "").strip()
    if not en:
        return zh
    if not zh:
        return en
    return f"{zh} / {en}"


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _name_keys(name_bi: str, name_zh: Optional[str] = None, name_en: Optional[str] = None) -> List[str]:
    keys = set()
    if name_bi:
        keys.add(_norm(name_bi))
        for part in re.split(r"[\/|,;（）()]+", name_bi):
            part = part.strip()
            if part:
                keys.add(_norm(part))
    for val in (name_zh, name_en):
        if val:
            keys.add(_norm(val))
    return [k for k in keys if k]


class KnowledgeGraph:
    def __init__(self) -> None:
        self.symptoms: Dict[str, Dict[str, Any]] = {}
        self.syndromes: Dict[str, Dict[str, Any]] = {}
        self._symptom_index: Dict[str, str] = {}
        self._syndrome_index: Dict[str, str] = {}
        self._symptom_counter = 0
        self._syndrome_counter = 0

    # ---- ID helpers ----
    def _new_symptom_id(self) -> str:
        self._symptom_counter += 1
        return f"symp-{self._symptom_counter:04d}"

    def _new_syndrome_id(self) -> str:
        self._syndrome_counter += 1
        return f"syn-{self._syndrome_counter:04d}"

    # ---- Index helpers ----
    def _index_symptom(self, sid: str, name: str, zh: str, en: str) -> None:
        for k in _name_keys(name, zh, en):
            self._symptom_index[k] = sid

    def _index_syndrome(self, sid: str, name: str, zh: str, en: str) -> None:
        for k in _name_keys(name, zh, en):
            self._syndrome_index[k] = sid

    def _reindex_symptom(self, sid: str) -> None:
        node = self.symptoms[sid]
        for k, v in list(self._symptom_index.items()):
            if v == sid:
                del self._symptom_index[k]
        self._index_symptom(sid, node["name"], node.get("name_zh", ""), node.get("name_en", ""))

    def _reindex_syndrome(self, sid: str) -> None:
        node = self.syndromes[sid]
        for k, v in list(self._syndrome_index.items()):
            if v == sid:
                del self._syndrome_index[k]
        self._index_syndrome(sid, node["name"], node.get("name_zh", ""), node.get("name_en", ""))

    # ---- Lookup ----
    def find_symptom_id(self, name_or_bi: str) -> Optional[str]:
        if not name_or_bi:
            return None
        key = _norm(name_or_bi)
        if key in self._symptom_index:
            return self._symptom_index[key]
        # Try parts
        for part in _name_keys(name_or_bi):
            sid = self._symptom_index.get(part)
            if sid:
                return sid
        return None

    def find_syndrome_id(self, name_or_bi: str) -> Optional[str]:
        if not name_or_bi:
            return None
        key = _norm(name_or_bi)
        if key in self._syndrome_index:
            return self._syndrome_index[key]
        for part in _name_keys(name_or_bi):
            sid = self._syndrome_index.get(part)
            if sid:
                return sid
        return None

    # ---- Mutation ----
    def add_symptom(self, name_zh: str, name_en: str, desc_zh: str, desc_en: str) -> Tuple[str, Dict[str, Any]]:
        name = bi(name_zh, name_en)
        # dedupe by name
        existing = self.find_symptom_id(name) or self.find_symptom_id(name_zh) or self.find_symptom_id(name_en)
        if existing:
            return existing, self.symptoms[existing]
        sid = self._new_symptom_id()
        node = {
            "id": sid,
            "type": "symptom",
            "name": name,
            "name_zh": name_zh,
            "name_en": name_en,
            "description": bi(desc_zh, desc_en),
            "syndromes": [],  # list of syndrome ids
        }
        self.symptoms[sid] = node
        self._index_symptom(sid, name, name_zh, name_en)
        return sid, node

    def add_syndrome(
        self,
        name_zh: str,
        name_en: str,
        desc_zh: str,
        desc_en: str,
        herbal_zh: str,
        herbal_en: str,
    ) -> Tuple[str, Dict[str, Any]]:
        name = bi(name_zh, name_en)
        existing = self.find_syndrome_id(name) or self.find_syndrome_id(name_zh) or self.find_syndrome_id(name_en)
        if existing:
            return existing, self.syndromes[existing]
        sid = self._new_syndrome_id()
        node = {
            "id": sid,
            "type": "syndrome",
            "name": name,
            "name_zh": name_zh,
            "name_en": name_en,
            "description": bi(desc_zh, desc_en),
            "symptoms": [],  # list of symptom ids
            "herbal_treatments": bi(herbal_zh, herbal_en),
        }
        self.syndromes[sid] = node
        self._index_syndrome(sid, name, name_zh, name_en)
        return sid, node

    def edit_symptom(self, target: str, name_zh: Optional[str] = None, name_en: Optional[str] = None,
                      desc_zh: Optional[str] = None, desc_en: Optional[str] = None) -> str:
        sid = self.find_symptom_id(target)
        if not sid:
            return f"WARNING: Symptom not found: {target}"
        node = self.symptoms[sid]
        if name_zh is not None or name_en is not None:
            node["name_zh"] = name_zh or node.get("name_zh", "")
            node["name_en"] = name_en or node.get("name_en", "")
            node["name"] = bi(node["name_zh"], node["name_en"])
            self._reindex_symptom(sid)
        if desc_zh is not None or desc_en is not None:
            zh = desc_zh or node.get("description", "").split(" / ")[0]
            en = desc_en or (node.get("description", "").split(" / ")[-1] if " / " in node.get("description", "") else "")
            node["description"] = bi(zh, en)
        return f"OK: Symptom updated: {node['name']}"

    def edit_syndrome(self, target: str, name_zh: Optional[str] = None, name_en: Optional[str] = None,
                      desc_zh: Optional[str] = None, desc_en: Optional[str] = None,
                      herbal_zh: Optional[str] = None, herbal_en: Optional[str] = None) -> str:
        sid = self.find_syndrome_id(target)
        if not sid:
            return f"WARNING: Syndrome not found: {target}"
        node = self.syndromes[sid]
        if name_zh is not None or name_en is not None:
            node["name_zh"] = name_zh or node.get("name_zh", "")
            node["name_en"] = name_en or node.get("name_en", "")
            node["name"] = bi(node["name_zh"], node["name_en"])
            self._reindex_syndrome(sid)
        if desc_zh is not None or desc_en is not None:
            zh = desc_zh or node.get("description", "").split(" / ")[0]
            en = desc_en or (node.get("description", "").split(" / ")[-1] if " / " in node.get("description", "") else "")
            node["description"] = bi(zh, en)
        if herbal_zh is not None or herbal_en is not None:
            # preserve old halves if missing
            old = node.get("herbal_treatments", "")
            old_zh = old.split(" / ")[0] if old else ""
            old_en = old.split(" / ")[-1] if (old and " / " in old) else ""
            node["herbal_treatments"] = bi(herbal_zh or old_zh, herbal_en or old_en)
        return f"OK: Syndrome updated: {node['name']}"

    def link(self, syndrome_name: str, symptom_name: str) -> str:
        syn_id = self.find_syndrome_id(syndrome_name)
        sym_id = self.find_symptom_id(symptom_name)
        if not syn_id and not sym_id:
            return f"WARNING: Neither found: syndrome={syndrome_name}, symptom={symptom_name}"
        if not syn_id:
            return f"WARNING: Syndrome not found: {syndrome_name}"
        if not sym_id:
            return f"WARNING: Symptom not found: {symptom_name}"
        syn = self.syndromes[syn_id]
        sym = self.symptoms[sym_id]
        if sym_id not in syn["symptoms"]:
            syn["symptoms"].append(sym_id)
        if syn_id not in sym["syndromes"]:
            sym["syndromes"].append(syn_id)
        return f"OK: Linked {syn['name']} <-> {sym['name']}"

    def split_symptom(self, original_name: str, children: List[Dict[str, str]]) -> str:
        sid = self.find_symptom_id(original_name)
        if not sid:
            return f"WARNING: Symptom not found: {original_name}"
        parent = self.symptoms[sid]
        # Copy links from parent to new children
        created = []
        for c in children:
            name_zh = c.get("name_zh", "").strip()
            name_en = c.get("name_en", "").strip()
            desc_zh = c.get("description_zh", "").strip()
            desc_en = c.get("description_en", "").strip()
            cid, child = self.add_symptom(name_zh, name_en, desc_zh, desc_en)
            # copy edges
            for syn_id in parent.get("syndromes", []):
                syn = self.syndromes[syn_id]
                if cid not in syn["symptoms"]:
                    syn["symptoms"].append(cid)
                if syn_id not in child["syndromes"]:
                    child["syndromes"].append(syn_id)
            created.append(child["name"])
        return f"OK: Split {parent['name']} into: {', '.join(created)}"

    # ---- Export / summary ----
    def lists_for_prompt(self) -> Tuple[List[str], List[str]]:
        symptom_names = [n["name"] for n in self.symptoms.values()]
        syndrome_names = [n["name"] for n in self.syndromes.values()]
        return sorted(symptom_names), sorted(syndrome_names)

    def to_json(self) -> Dict[str, Any]:
        return {
            "symptoms": list(self.symptoms.values()),
            "syndromes": list(self.syndromes.values()),
        }


class AgentState:
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.current_batch_text: str = ""


def save_graph_and_checkpoint(
    *,
    graph: KnowledgeGraph,
    next_chapter_idx: int,
    chapters: List[Dict[str, Any]],
    total_chapters: int,
    checkpoint_path: str,
    checkpoint_dir: str,
) -> str:
    """Persist graph.json and both latest and timestamped checkpoints. Returns the timestamped path."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    summary = graph.to_json()
    with open("graph.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_payload = {
        "next_chapter_idx": next_chapter_idx,
        "graph": summary,
        "book_meta": {
            "chapters_total": total_chapters,
            "chapters": [{"title": ch["title"]} for ch in chapters],
        },
        "updated_at": stamp,
    }

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(latest_payload, f, ensure_ascii=False, indent=2)

    ts_path = os.path.join(checkpoint_dir, f"checkpoint_{stamp}.json")
    with open(ts_path, "w", encoding="utf-8") as f:
        json.dump(latest_payload, f, ensure_ascii=False, indent=2)

    return ts_path


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
            delay = 2 ** attempt
            print(f"[retry] {type(e).__name__}: sleeping {delay}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
            attempt += 1


# =============================
# Book parsing
# =============================


def parse_book(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"book.txt not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by chapters and sections. We keep titles for context.
    sections: List[Dict[str, str]] = []
    chapter_blocks = re.split(r"^#CHAPTER\s*" , content, flags=re.MULTILINE)
    # The first split part may be preface text; ignore unless it has content.
    for block in chapter_blocks:
        block = block.strip()
        if not block:
            continue
        # Chapter title is first line; rest is content
        lines = block.splitlines()
        chapter_title = lines[0].strip()
        rest = "\n".join(lines[1:])

        # Split sections within chapter
        sect_blocks = re.split(r"^#SECTION\s*", rest, flags=re.MULTILINE)
        if len(sect_blocks) <= 1:
            # Treat entire chapter as one section if no #SECTION markers
            sections.append({
                "chapter": chapter_title,
                "section": "(whole chapter)",
                "text": rest.strip(),
            })
        else:
            for sblock in sect_blocks:
                sblock = sblock.strip()
                if not sblock:
                    continue
                slines = sblock.splitlines()
                section_title = slines[0].strip()
                stext = "\n".join(slines[1:]).strip()
                sections.append({
                    "chapter": chapter_title,
                    "section": section_title,
                    "text": stext,
                })
    return sections


def format_sections_for_prompt(sections: List[Dict[str, str]]) -> str:
    parts = []
    for i, sec in enumerate(sections, 1):
        header = f"[Section {i}] CHAPTER: {sec['chapter']} | SECTION: {sec['section']}"
        parts.append(header)
        parts.append(sec["text"].strip())
    return "\n\n".join(parts)


def parse_book_grouped(path: str) -> List[Dict[str, Any]]:
    """Return a list of chapters with titles and their sections (chapter-bounded).

    Each chapter: {"title": str, "sections": [ {chapter, section, text}, ... ]}
    """
    chapters: List[Dict[str, Any]] = []
    flat = parse_book(path)
    by_title: Dict[str, List[Dict[str, str]]] = {}
    for sec in flat:
        by_title.setdefault(sec["chapter"], []).append(sec)
    for title, sects in by_title.items():
        chapters.append({"title": title, "sections": sects})
    return chapters


# =============================
# Tooling: schemas and dispatch
# =============================


def build_tools_schema() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "add_symptom_node",
                "description": "Add a Symptom node with bilingual name and description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name_zh": {"type": "string"},
                        "name_en": {"type": "string"},
                        "description_zh": {"type": "string"},
                        "description_en": {"type": "string"},
                    },
                    "required": ["name_zh", "name_en", "description_zh", "description_en"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "add_syndrome_node",
                "description": "Add a Syndrome node with bilingual name, description, and herbal treatments.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name_zh": {"type": "string"},
                        "name_en": {"type": "string"},
                        "description_zh": {"type": "string"},
                        "description_en": {"type": "string"},
                        "herbal_zh": {"type": "string"},
                        "herbal_en": {"type": "string"},
                    },
                    "required": [
                        "name_zh",
                        "name_en",
                        "description_zh",
                        "description_en",
                        "herbal_zh",
                        "herbal_en",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "link_syndrome_symptom",
                "description": "Create a bidirectional edge between a syndrome and a symptom by name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "syndrome_name": {"type": "string"},
                        "symptom_name": {"type": "string"},
                    },
                    "required": ["syndrome_name", "symptom_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "split_symptom_node",
                "description": "Split a symptom into finer-grained child symptoms; copies existing syndrome links.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "original_name": {"type": "string"},
                        "children": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name_zh": {"type": "string"},
                                    "name_en": {"type": "string"},
                                    "description_zh": {"type": "string"},
                                    "description_en": {"type": "string"},
                                },
                                "required": ["name_zh", "name_en", "description_zh", "description_en"],
                            },
                        },
                    },
                    "required": ["original_name", "children"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_symptom_node",
                "description": "Edit a symptom's bilingual name and/or description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_name": {"type": "string"},
                        "new_name_zh": {"type": "string"},
                        "new_name_en": {"type": "string"},
                        "description_zh": {"type": "string"},
                        "description_en": {"type": "string"},
                    },
                    "required": ["target_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_syndrome_node",
                "description": "Edit a syndrome's bilingual name, description, and herbal treatments.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_name": {"type": "string"},
                        "new_name_zh": {"type": "string"},
                        "new_name_en": {"type": "string"},
                        "description_zh": {"type": "string"},
                        "description_en": {"type": "string"},
                        "herbal_zh": {"type": "string"},
                        "herbal_en": {"type": "string"},
                    },
                    "required": ["target_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "expand_symptom",
                "description": "Use the current batch's text to expand a symptom's bilingual description. Warn if not found.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_name": {"type": "string"},
                        "notes": {"type": "string", "description": "Optional hints or focus."},
                    },
                    "required": ["target_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "expand_syndrome",
                "description": "Use the current batch's text to expand a syndrome's bilingual description and herbal treatments. Warn if not found.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_name": {"type": "string"},
                        "notes": {"type": "string", "description": "Optional hints or focus."},
                    },
                    "required": ["target_name"],
                },
            },
        },
    ]


# Tool implementations will capture state via closures set later
def make_tool_dispatch(state: AgentState):
    kg = state.graph

    def add_symptom_node(args: Dict[str, Any]) -> str:
        sid, node = kg.add_symptom(
            args.get("name_zh", ""),
            args.get("name_en", ""),
            args.get("description_zh", ""),
            args.get("description_en", ""),
        )
        return json.dumps({"status": "ok", "symptom_id": sid, "name": node["name"]}, ensure_ascii=False)

    def add_syndrome_node(args: Dict[str, Any]) -> str:
        sid, node = kg.add_syndrome(
            args.get("name_zh", ""),
            args.get("name_en", ""),
            args.get("description_zh", ""),
            args.get("description_en", ""),
            args.get("herbal_zh", ""),
            args.get("herbal_en", ""),
        )
        return json.dumps({"status": "ok", "syndrome_id": sid, "name": node["name"]}, ensure_ascii=False)

    def link_syndrome_symptom(args: Dict[str, Any]) -> str:
        msg = kg.link(args.get("syndrome_name", ""), args.get("symptom_name", ""))
        return msg

    def split_symptom_node(args: Dict[str, Any]) -> str:
        return kg.split_symptom(args.get("original_name", ""), args.get("children", []) or [])

    def edit_symptom_node(args: Dict[str, Any]) -> str:
        return kg.edit_symptom(
            args.get("target_name", ""),
            name_zh=args.get("new_name_zh"),
            name_en=args.get("new_name_en"),
            desc_zh=args.get("description_zh"),
            desc_en=args.get("description_en"),
        )

    def edit_syndrome_node(args: Dict[str, Any]) -> str:
        return kg.edit_syndrome(
            args.get("target_name", ""),
            name_zh=args.get("new_name_zh"),
            name_en=args.get("new_name_en"),
            desc_zh=args.get("description_zh"),
            desc_en=args.get("description_en"),
            herbal_zh=args.get("herbal_zh"),
            herbal_en=args.get("herbal_en"),
        )

    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except Exception:
            # Try to extract a JSON object substring
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
            return None

    def expand_symptom(args: Dict[str, Any]) -> str:
        target = args.get("target_name", "")
        sid = kg.find_symptom_id(target)
        if not sid:
            return f"WARNING: Symptom not found: {target}"
        node = kg.symptoms[sid]
        prompt = [
            {
                "role": "system",
                "content": (
                    "You expand Traditional Chinese Medicine symptom descriptions. "
                    "Output compact JSON only with keys: description_zh, description_en. "
                    "Chinese first, then faithful English. If translation is unknown, use pinyin (no hyphens)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Symptom name: {node['name']}\n"
                    f"Current description: {node.get('description','')}\n"
                    f"Context from book (batch):\n{state.current_batch_text}\n\n"
                    f"Notes: {args.get('notes','')}\n"
                    "Respond with strictly JSON."
                ),
            },
        ]
        resp = chat_with_retry(prompt)
        text = first_message_text(resp)
        data = _extract_json(text) or {}
        desc_zh = data.get("description_zh") or node.get("description", "").split(" / ")[0]
        desc_en = data.get("description_en") or (node.get("description", "").split(" / ")[-1] if " / " in node.get("description", "") else "")
        node["description"] = bi(desc_zh, desc_en)
        return f"OK: Expanded symptom {node['name']}"

    def expand_syndrome(args: Dict[str, Any]) -> str:
        target = args.get("target_name", "")
        sid = kg.find_syndrome_id(target)
        if not sid:
            return f"WARNING: Syndrome not found: {target}"
        node = kg.syndromes[sid]
        prompt = [
            {
                "role": "system",
                "content": (
                    "You expand TCM syndrome descriptions and herbal treatments. "
                    "Output compact JSON only with keys: description_zh, description_en, herbal_zh, herbal_en. "
                    "Chinese first, then faithful English. If translation is unknown, use pinyin (no hyphens)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Syndrome name: {node['name']}\n"
                    f"Current description: {node.get('description','')}\n"
                    f"Current herbal treatments: {node.get('herbal_treatments','')}\n"
                    f"Context from book (batch):\n{state.current_batch_text}\n\n"
                    f"Notes: {args.get('notes','')}\n"
                    "Respond with strictly JSON."
                ),
            },
        ]
        resp = chat_with_retry(prompt)
        text = first_message_text(resp)
        data = _extract_json(text) or {}
        desc_zh = data.get("description_zh") or node.get("description", "").split(" / ")[0]
        desc_en = data.get("description_en") or (node.get("description", "").split(" / ")[-1] if " / " in node.get("description", "") else "")
        herb_zh = data.get("herbal_zh") or node.get("herbal_treatments", "").split(" / ")[0]
        herb_en = data.get("herbal_en") or (node.get("herbal_treatments", "").split(" / ")[-1] if " / " in node.get("herbal_treatments", "") else "")
        node["description"] = bi(desc_zh, desc_en)
        node["herbal_treatments"] = bi(herb_zh, herb_en)
        return f"OK: Expanded syndrome {node['name']}"

    return {
        "add_symptom_node": add_symptom_node,
        "add_syndrome_node": add_syndrome_node,
        "link_syndrome_symptom": link_syndrome_symptom,
        "split_symptom_node": split_symptom_node,
        "edit_symptom_node": edit_symptom_node,
        "edit_syndrome_node": edit_syndrome_node,
        "expand_symptom": expand_symptom,
        "expand_syndrome": expand_syndrome,
    }


# =============================
# Agent loop
# =============================


SYSTEM_INSTRUCTIONS = (
    "You are building a knowledge graph for a short classic Chinese medicine book. "
    "Always operate via function tools only to add/edit/link/split/expand nodes. "
    "Do not answer with prose unless there is nothing to do.\n\n"
    "Data model: Two node types.\n"
    "- Symptom: fields -> name (Chinese first + English), description (Chinese + English), syndrome pointers.\n"
    "- Syndrome: fields -> name (Chinese first + English), description/insights (Chinese + English), symptoms, herbal treatments (Chinese + English).\n\n"
    "Rules:\n"
    "- Treat diagnostic criteria and signs as symptoms too (e.g., 脉象 qualities like 强脉/弱脉; 舌象/苔象; 面色; 口渴/汗出). Each distinct quality is its own Symptom node (e.g., 强脉 / strong pulse and 弱脉 / weak pulse are separate).\n"
    "- For every field, include Chinese first then English with the same meaning; if you cannot translate, use pinyin without hyphens.\n"
    "- Use commonly understood English canonical symptom names (e.g., 'coughing') and accurate bilingual syndrome names.\n"
    "- Ignore musings, anecdotes, or unrelated commentary unless it can be used to usefully modify the graph without breaking the schema.\n"
    "- Link syndromes <-> symptoms bidirectionally using the link tool.\n"
    "- Use split_symptom_node for finer granularity (e.g., cough -> wet cough, dry cough) and then add any additional links.\n"
    "- Use expand_symptom and expand_syndrome to enrich descriptions (and herbs for syndromes) based on current batch context; these tools warn if target not found.\n"
    "- Avoid duplicate nodes; reuse existing names when present.\n"
    "- Work incrementally; multiple tool calls are expected per batch.\n"
    "- ONLY make calls to expand at the start of a session or otherwise necessary, there is no need to verify your changes at the very end of your session thats too expensive"
    "- ONLY links syndromes and symptoms mentioned in the current chapter. Assume that connections mentioned in the description fields of symptoms and syndromes are already linked"
    "- When writing descriptions of symptoms and syndromes, be concise, 1-2 sentences per language max"
)


def names_memory(graph: KnowledgeGraph) -> str:
    sym_names, syn_names = graph.lists_for_prompt()
    return (
        "Existing Symptom Names (bilingual):\n- "
        + "\n- ".join(sym_names or ["<none>"])
        + "\n\nExisting Syndrome Names (bilingual):\n- "
        + "\n- ".join(syn_names or ["<none>"])
    )


def run_kg_builder(
    *,
    batch_size: int = 10,
    max_tool_rounds: int = 8,
    chapters_to_process: Optional[int] = None,
    checkpoint_path: str = "checkpoints/checkpoint_latest.json",
    checkpoint_dir: str = "checkpoints",
    reset: bool = False,
) -> None:
    # Load or initialize graph and progress
    chapters = parse_book_grouped("book.txt")
    total_chapters = len(chapters)

    graph = KnowledgeGraph()
    next_chapter_idx = 0

    # Load checkpoint if present
    if not reset and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            # Restore graph
            data = ckpt.get("graph", {})
            if data:
                graph.symptoms = {n["id"]: n for n in data.get("symptoms", [])}
                graph.syndromes = {n["id"]: n for n in data.get("syndromes", [])}
                # Rebuild indices and counters
                graph._symptom_index.clear()
                graph._syndrome_index.clear()
                for n in graph.symptoms.values():
                    graph._index_symptom(n["id"], n.get("name", ""), n.get("name_zh", ""), n.get("name_en", ""))
                for n in graph.syndromes.values():
                    graph._index_syndrome(n["id"], n.get("name", ""), n.get("name_zh", ""), n.get("name_en", ""))
                # Update counters based on ids
                def _max_id_num(items: Dict[str, Any], prefix: str) -> int:
                    m = 0
                    for k in items.keys():
                        try:
                            num = int(k.replace(prefix, "").replace("-", ""))
                            m = max(m, num)
                        except Exception:
                            continue
                    return m
                graph._symptom_counter = max(_max_id_num(graph.symptoms, "symp"), len(graph.symptoms))
                graph._syndrome_counter = max(_max_id_num(graph.syndromes, "syn"), len(graph.syndromes))
            next_chapter_idx = int(ckpt.get("next_chapter_idx", 0))
        except Exception:
            next_chapter_idx = 0

    state = AgentState(graph)
    tools_schema = build_tools_schema()
    dispatch = make_tool_dispatch(state)

    # Determine number of chapters to process this run
    remaining = total_chapters - next_chapter_idx
    to_process = remaining if (chapters_to_process is None or chapters_to_process <= 0) else min(remaining, chapters_to_process)

    # Iterate chapters and within each, process in section-batches
    for ch_idx in range(next_chapter_idx, next_chapter_idx + to_process):
        chapter = chapters[ch_idx]
        sects = chapter["sections"]
        for s in range(0, len(sects), batch_size):
            batch = sects[s : s + batch_size]
            state.current_batch_text = format_sections_for_prompt(batch)

            # Minimal progress print: chapter and section indices
            batch_end = min(s + batch_size, len(sects))
            print(f"[batch] ch {ch_idx + 1}/{total_chapters} sects {s + 1}-{batch_end} of {len(sects)}")

            # Reset context window except names list and system instructions
            msgs: List[Dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {
                    "role": "user",
                    "content": (
                        names_memory(graph)
                        + "\n\nYou are given the next batch of book sections from this chapter. "
                        + "Create or refine nodes and edges using tools only. "
                        + "When expanding, call expand_symptom/expand_syndrome; they warn if name not found.\n\n"
                        + state.current_batch_text
                    ),
                },
            ]

            rounds = 0
            while rounds < max_tool_rounds:
                resp = chat_with_retry(msgs, tools=tools_schema, tool_choice="auto")
                assistant_msg = resp.choices[0].message
                msgs.append(
                    {
                        "role": "assistant",
                        "content": assistant_msg.content or "",
                        "tool_calls": [tc.model_dump() for tc in (assistant_msg.tool_calls or [])],
                    }
                )

                if not assistant_msg.tool_calls:
                    break

                # Execute tool calls immediately after assistant tool_calls
                for tc in (assistant_msg.tool_calls or []):
                    name = tc.function.name
                    raw_args = tc.function.arguments or "{}"
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {}
                    tool_fn = dispatch.get(name)
                    if not tool_fn:
                        result = f"WARNING: Tool not implemented: {name}"
                    else:
                        # Minimal call print: tool and target label
                        def _tool_label(tname: str, a: Dict[str, Any]) -> str:
                            try:
                                if tname == "add_symptom_node":
                                    return bi(a.get("name_zh", ""), a.get("name_en", ""))
                                if tname == "add_syndrome_node":
                                    return bi(a.get("name_zh", ""), a.get("name_en", ""))
                                if tname == "edit_symptom_node":
                                    return a.get("target_name", "")
                                if tname == "edit_syndrome_node":
                                    return a.get("target_name", "")
                                if tname == "expand_symptom":
                                    return a.get("target_name", "")
                                if tname == "expand_syndrome":
                                    return a.get("target_name", "")
                                if tname == "link_syndrome_symptom":
                                    return f"{a.get('syndrome_name','')} <-> {a.get('symptom_name','')}"
                                if tname == "split_symptom_node":
                                    chn = a.get("children") or []
                                    return f"{a.get('original_name','')} -> {len(chn)}"
                            except Exception:
                                pass
                            return ""

                        label = _tool_label(name, args)
                        if label:
                            print(f"[call] {name}: {label}")
                        else:
                            print(f"[call] {name}")
                        try:
                            result = tool_fn(args)
                        except Exception as e:  # safety against tool crashes
                            result = f"ERROR: Tool {name} failed: {e}"
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

            # Pause briefly between batches to reduce request pressure
            # (sleep even if no tool calls to ensure consistent pacing)
            time.sleep(1)

        # Autosave after finishing this chapter
        next_chapter_idx = ch_idx + 1
        ts_path = save_graph_and_checkpoint(
            graph=graph,
            next_chapter_idx=next_chapter_idx,
            chapters=chapters,
            total_chapters=total_chapters,
            checkpoint_path=checkpoint_path,
            checkpoint_dir=checkpoint_dir,
        )
        print(f"[save] chapter {next_chapter_idx}/{total_chapters} checkpoint -> {ts_path}")

    # Update progress
    next_chapter_idx += to_process

    # Final save at end of run
    ts_path = save_graph_and_checkpoint(
        graph=graph,
        next_chapter_idx=next_chapter_idx,
        chapters=chapters,
        total_chapters=total_chapters,
        checkpoint_path=checkpoint_path,
        checkpoint_dir=checkpoint_dir,
    )

    # Print concise counts
    print(
        json.dumps(
            {
                "symptom_count": len(graph.symptoms),
                "syndrome_count": len(graph.syndromes),
                "edges": sum(len(s["symptoms"]) for s in graph.syndromes.values()),
                "next_chapter_idx": next_chapter_idx,
                "chapters_total": total_chapters,
                "checkpoint_latest": checkpoint_path,
                "checkpoint_saved": ts_path,
                "graph_saved": "graph.json",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build TCM knowledge graph from book.txt")
    parser.add_argument("--batch-size", type=int, default=10, help="Max sections per batch (within a chapter)")
    parser.add_argument("--max-tool-rounds", type=int, default=50, help="Max tool-call rounds per batch")
    parser.add_argument("--chapters", type=int, default=0, help="Number of chapters to process this run (0 = all remaining)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_latest.json", help="Path to latest checkpoint file")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to store timestamped checkpoints")
    parser.add_argument("--reset", action="store_true", help="Ignore existing checkpoint and start from chapter 0")

    args = parser.parse_args()

    run_kg_builder(
        batch_size=args.batch_size,
        max_tool_rounds=args.max_tool_rounds,
        chapters_to_process=args.chapters,
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        reset=args.reset,
    )
