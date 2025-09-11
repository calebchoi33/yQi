import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from endpoint import call_chat, first_message_text


def chat_with_retry(
    messages: List[Dict[str, Any]],
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = "auto",
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = 3,
):
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


def load_graph(path: str = "graph.json") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph not found at {path}. Run agent.py first to build it.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class KB:
    def __init__(self, data: Dict[str, Any]):
        self.symptoms_by_id: Dict[str, Dict[str, Any]] = {}
        self.syndromes_by_id: Dict[str, Dict[str, Any]] = {}
        for n in data.get("symptoms", []):
            self.symptoms_by_id[n["id"]] = n
        for n in data.get("syndromes", []):
            self.syndromes_by_id[n["id"]] = n

    def symptom_catalog(self) -> str:
        lines = []
        for sid, node in sorted(self.symptoms_by_id.items()):
            lines.append(f"{sid} - {node.get('name','')}")
        return "\n".join(lines)

    def expand_symptom(self, symptom_id: str) -> str:
        node = self.symptoms_by_id.get(symptom_id)
        if not node:
            return f"WARNING: Symptom id not found: {symptom_id}"
        return json.dumps(
            {
                "symptom_id": node["id"],
                "name": node.get("name", ""),
                "description": node.get("description", ""),
                "syndromes": node.get("syndromes", []),
            },
            ensure_ascii=False,
        )

    def score_syndromes(self, symptom_ids: List[str]) -> List[Tuple[str, float, List[str]]]:
        # Weight earlier symptoms higher: w = 1.0 / (index+1)
        weights = {sid: 1.0 / (i + 1) for i, sid in enumerate(symptom_ids)}
        results: List[Tuple[str, float, List[str]]] = []
        for syn_id, syn in self.syndromes_by_id.items():
            syn_syms = syn.get("symptoms", [])
            matched = [sid for sid in symptom_ids if sid in syn_syms]
            if not matched:
                continue
            score = sum(weights.get(sid, 0.0) for sid in matched)
            # slight preference for syndromes that are more specific (fewer total symptoms)
            total_sym = max(len(syn_syms), 1)
            score = score + 0.01 * (1.0 / total_sym)
            results.append((syn_id, score, matched))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def search_syndromes(self, symptom_ids: List[str]) -> str:
        # Validate ids and collect warnings
        warnings: List[str] = []
        valid_ids: List[str] = []
        for sid in symptom_ids:
            if sid in self.symptoms_by_id:
                valid_ids.append(sid)
            else:
                warnings.append(f"WARNING: Symptom id not found: {sid}")
        if not valid_ids:
            return "WARNING: No valid symptom ids provided."
        scored = self.score_syndromes(valid_ids)
        if not scored:
            return "WARNING: No matching syndromes for the given symptoms."
        top = scored[:10]
        payload = {
            "query_symptom_ids": valid_ids,
            "results": [],
        }
        for syn_id, score, match in top:
            syn = self.syndromes_by_id[syn_id]
            # Build full symptom listing for this syndrome
            sym_items = []
            for sid in syn.get("symptoms", []):
                sym = self.symptoms_by_id.get(sid)
                if sym:
                    sym_items.append({"id": sid, "name": sym.get("name", "")})
                else:
                    sym_items.append({"id": sid, "name": "<missing>"})
            payload["results"].append(
                {
                    "syndrome_id": syn_id,
                    "name": syn.get("name", ""),
                    "description": syn.get("description", ""),
                    "symptoms": sym_items,
                    "score": round(score, 4),
                    "matched_symptom_ids": match,
                }
            )
        if warnings:
            payload["warnings"] = warnings
        return json.dumps(payload, ensure_ascii=False)

    def give_diagnosis(self, syndrome_id: str) -> str:
        syn = self.syndromes_by_id.get(syndrome_id)
        if not syn:
            return f"WARNING: Syndrome id not found: {syndrome_id}"
        return json.dumps(
            {
                "syndrome_id": syn["id"],
                "name": syn.get("name", ""),
                "description": syn.get("description", ""),
                "herbal_treatments": syn.get("herbal_treatments", ""),
                "symptom_ids": syn.get("symptoms", []),
            },
            ensure_ascii=False,
        )


def build_tools_schema() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "ask_patient",
                "description": "Ask a specific, non-generic patient question; reference concrete symptoms by name/description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "symptom_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["question"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "expand_symptom",
                "description": "Return the bilingual description and metadata of a symptom by id. Warn if not found.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symptom_id": {"type": "string"},
                    },
                    "required": ["symptom_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_syndromes",
                "description": "Return top 10 likely syndromes given a list of symptom ids (sorted by patient bother/match). Include each syndrome's full description and all its symptoms (id + name). Warn on invalid ids or no matches.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symptom_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["symptom_ids"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "give_diagnosis",
                "description": "Return the bilingual description and herbal treatments for a chosen syndrome id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "syndrome_id": {"type": "string"},
                        "notes": {"type": "string", "description": "Optional rationale or instructions."},
                    },
                    "required": ["syndrome_id"],
                },
            },
        },
    ]


def make_tool_dispatch(kb: KB):
    def ask_patient(args: Dict[str, Any]) -> str:
        question = (args.get("question", "") or "").strip()
        ids = args.get("symptom_ids", []) or []
        return json.dumps({"question": question, "symptom_ids": ids}, ensure_ascii=False)

    def expand_symptom(args: Dict[str, Any]) -> str:
        return kb.expand_symptom(args.get("symptom_id", ""))

    def search_syndromes(args: Dict[str, Any]) -> str:
        return kb.search_syndromes(args.get("symptom_ids", []) or [])

    def give_diagnosis(args: Dict[str, Any]) -> str:
        return kb.give_diagnosis(args.get("syndrome_id", ""))

    return {
        "ask_patient": ask_patient,
        "expand_symptom": expand_symptom,
        "search_syndromes": search_syndromes,
        "give_diagnosis": give_diagnosis,
    }


SYSTEM = (
    "You are a TCM triage chatbot using a knowledge graph of symptoms and syndromes.\n"
    "1) Use expand_symptom(symptom_id) to clarify whether their descriptions match specific symptoms.\n"
    "2) Use search_syndromes(list[symptom_ids]) to find likely syndromes (include full descriptions and all symptoms).\n"
    "3) Ask clarifying questions with ask_patient(question, symptom_ids), referencing concrete symptoms by name/description; never generic.\n"
    "4) Once confident, call give_diagnosis(syndrome_id) to present the likely syndrome and herbal treatments.\n\n"
    "Rules:\n"
    "- Do not output plain text responses; communicate only via tool calls.\n"
    "- Prefer concise, clear, patient-friendly questions and never generic; after the first symptom, always mention specific symptoms to check.\n"
    "- Use search_syndromes often to propose the most probable syndromes, then ask about their key symptoms via ask_patient.\n"
    "- Never call expand_symptom and search_syndromes in the same turn; pick the most useful one.\n"
    "- Syndromes need not exactly match; use best judgement.\n"
    "- If a tool returns a WARNING, acknowledge it and adjust.\n"
    "- Make only ONE tool call at a time. If you need multiple, choose the best next one, and wait for that one to return first.\n"
    "- After 30 total tool calls, you MUST call give_diagnosis with your best candidate.\n"
    "- The patient DOES not know about your knowledge graph, do not reference symptom IDs, just describe them by name and description."
    "- Determine the patient's preferred language from their first message, and respond in that language"
)


def symptom_catalog_text(kb: KB) -> str:
    return "Symptom Catalog (id - name):\n" + kb.symptom_catalog()


def run_chat():
    kb = KB(load_graph("graph.json"))
    tools = build_tools_schema()
    dispatch = make_tool_dispatch(kb)

    # Show catalog to the patient; initialize agent only after first user input
    print("Chatbot is ready. Type your messages; Ctrl+C to exit.\n")
    catalog = symptom_catalog_text(kb)

    print("Describe your symptoms.\n")

    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "system", "content": catalog},
    ]

    tool_calls_count = 0
    diagnosis_done = False

    while True:
        try:
            user_msg = input("You: ")
        except KeyboardInterrupt:
            print("\nExiting.")
            return

        msgs.append({"role": "user", "content": user_msg})

        # Blocking loop: keep invoking model until it asks user (ask_patient) or returns diagnosis
        while True:
            if tool_calls_count >= 30 and not diagnosis_done:
                msgs.append({
                    "role": "system",
                    "content": "Tool budget reached (30). You MUST call give_diagnosis now with your best candidate.",
                })

            resp = chat_with_retry(msgs, tools=tools, tool_choice="auto")
            assistant_msg = resp.choices[0].message
            entry: Dict[str, Any] = {"role": "assistant"}
            if assistant_msg.tool_calls:
                entry["tool_calls"] = [tc.model_dump() for tc in assistant_msg.tool_calls]
            if assistant_msg.content:
                entry["content"] = assistant_msg.content  # not shown to user
            msgs.append(entry)

            if assistant_msg.tool_calls:
                # Enforce single tool call per turn but respond to all tool_call_ids
                tcs = list(assistant_msg.tool_calls)
                tc = tcs[0]
                name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}

                # Print concise label
                def _label(tname: str, params: Dict[str, Any]) -> str:
                    try:
                        if tname == "expand_symptom":
                            sid = params.get("symptom_id", "")
                            nm = kb.symptoms_by_id.get(sid, {}).get("name", "")
                            return f"{sid} - {nm}"
                        if tname == "search_syndromes":
                            ids = params.get("symptom_ids", []) or []
                            return f"{len(ids)} symptoms"
                        if tname == "give_diagnosis":
                            syn_id = params.get("syndrome_id", "")
                            nm = kb.syndromes_by_id.get(syn_id, {}).get("name", "")
                            return f"{syn_id} - {nm}"
                        if tname == "ask_patient":
                            return ""
                    except Exception:
                        pass
                    return ""

                label = _label(name, args)
                print(f"[call] {name}: {label}" if label else f"[call] {name}")

                tool_fn = dispatch.get(name)
                if not tool_fn:
                    result = f"WARNING: Tool not implemented: {name}"
                else:
                    try:
                        result = tool_fn(args)
                    except Exception as e:
                        result = f"ERROR: Tool {name} failed: {e}"

                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": str(result),
                })
                tool_calls_count += 1

                # Print warnings/errors from tool
                if isinstance(result, str):
                    if result.startswith("WARNING"):
                        print(f"[warn] {result}")
                    elif result.startswith("ERROR"):
                        print(f"[error] {result}")

                if name == "ask_patient":
                    # Surface question to user and break to wait for input
                    try:
                        payload = json.loads(result) if isinstance(result, str) else {}
                    except Exception:
                        payload = {}
                    question = (payload.get("question") or "").strip()
                    if question:
                        print(f"Assistant: {question}\n")
                    # Also respond to any extra tool calls with warnings to satisfy API contract
                    for extra in tcs[1:]:
                        extra_name = extra.function.name
                        warn_msg = f"WARNING: Only one tool call allowed at a time. Ignored: {extra_name}"
                        print(f"[warn] {warn_msg}")
                        msgs.append({
                            "role": "tool",
                            "tool_call_id": extra.id,
                            "name": extra_name,
                            "content": warn_msg,
                        })
                    break  # back to outer loop for user input

                if name == "give_diagnosis" and isinstance(result, str) and not result.startswith("WARNING"):
                    try:
                        diag = json.loads(result)
                        print("Diagnosis: " + (diag.get("name", "") or ""))
                        if diag.get("description"):
                            print("Description: " + diag["description"])
                        if diag.get("herbal_treatments"):
                            print("Herbal treatments: " + diag["herbal_treatments"]) 
                        print("")
                    except Exception:
                        print(result)
                    print("Diagnosis complete. Goodbye.")
                    return

                # For expand/search, loop to let the model react to tool output immediately
                # Before looping, emit warnings for any extra tool calls to satisfy API contract
                for extra in tcs[1:]:
                    extra_name = extra.function.name
                    warn_msg = f"WARNING: Only one tool call allowed at a time. Ignored: {extra_name}"
                    print(f"[warn] {warn_msg}")
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": extra.id,
                        "name": extra_name,
                        "content": warn_msg,
                    })
                continue

            # No tool calls: nudge to use a single tool and loop again (blocking)
            msgs.append({
                "role": "system",
                "content": (
                    "Reminder: communicate only via a SINGLE tool call. Use ask_patient to ask specific questions with symptom names/ids, or search_syndromes to guide next questions."
                ),
            })
            continue


if __name__ == "__main__":
    run_chat()
