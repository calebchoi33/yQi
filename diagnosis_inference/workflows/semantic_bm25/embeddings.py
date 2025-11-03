"""Embedding utilities and per-term tag extraction for Semantic BM25 workflow."""

import time
import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
_CLIENT: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Return a singleton OpenAI client using OPENAI_API_KEY."""
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ["OPENAI_API_KEY"]
        _CLIENT = OpenAI(api_key=api_key)
    return _CLIENT


def call_chat(
    messages: List[Dict[str, Any]],
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = "auto",
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> Any:
    """
    Call the Chat Completions API with optional tools support.

    Args:
        messages: List of role/content dicts per Chat Completions spec.
        tools: Optional list of tool schemas (function tools).
        tool_choice: Tool choice policy (e.g., "auto" or specific function name).
        model: Model name; defaults to env OPENAI_MODEL or 'gpt-4o-mini'.
        temperature: Sampling temperature.

    Returns:
        The API response object as returned by the SDK.
    """

    client = _get_client()
    model = model or os.environ["OPENAI_MODEL"]

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if tools:
        kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

    return client.chat.completions.create(**kwargs)


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
                raise e

            delay = 2**attempt

            print(f"--------------------------------")
            print(f"Error: {e}")
            print(f"Attempt {attempt + 1}/{max_retries}")
            print(f"Retrying in {delay} seconds...")
            print(f"--------------------------------")

            time.sleep(delay)
            attempt += 1


def _normalize(vectors):
    """
    Normalize the embeddings.
    """
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def get_batch_embeddings(symptoms: List[str]):
    """
    Embeds all symptoms. Returns a dict of symptom -> symptom_embed
    """

    # get embeddings
    client = _get_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=symptoms)
    embeddings = [item.embedding for item in response.data]
    
    # normalize
    normalized_embeddings = _normalize(embeddings)
    
    # convert to dict
    res = {}
    for i in range(len(symptoms)):
        res[symptoms[i]] = normalized_embeddings[i]
    
    return res


def extract_semantic_symptoms_and_freq(text: str) -> Dict[str, int]:
    """
    Takes in a section and uses an LLM to extract symptoms with the frequency they occur in the text.
    Returns a dictionary of all symptoms and their corresponding frequencies in the section.
    """

    system_instructions = (
        "You are an expert Traditional Chinese Medicine assistant that analyzes TCM textbook sections written in Chinese. "
        "Your task is to identify and extract all symptoms mentioned within a textbook section, along with the frequency that that symptom is mentioned within the section.\n\n"
        "INSTRUCTIONS:\n"
        "- Carefully read the text section and extract every symptom mentioned, even if expressed indirectly.\n"
        "- For each symptom, normalize it into its standard or most widely recognized form, counting synonyms and paraphrases as a mention of the same symptom.\n"
        "- Keep track of the number of times each symptom is mentioned within the text, including synonyms and paraphrases.\n"
        "- Return your results by calling the provided 'symptoms_with_frequencies_mentioned' tool.\n"
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "symptoms_with_frequencies_mentioned",
                "description": "Return all TCM symptoms expressed in the section, standardized to their common textbook forms, counting the number of occurences (frequency).",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symptoms_mentioned": {
                            "type": "array",
                            "description": "List of standardized TCM symptoms expressed in the query and frequency in the section.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "symptom": {
                                        "type": "string",
                                        "description": "A symptom in its normalized or commonly known TCM form.",
                                    },
                                    "frequency": {
                                        "type": "integer",
                                        "description": "The number of times the symptom and synonyms/paraphrases were mentioned in the section.",
                                    },
                                },
                                "required": ["symptom", "frequency"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["symptoms_mentioned"],
                    "additionalProperties": False,
                }
            },
        }
    ]

    input_msg = [
        {"role": "system", "content": system_instructions},
        {
            "role": "user",
            "content": (
                "Analyze the following Traditional Chinese Medicine textbook section and extract all symptoms mentioned and its frequency in the text.\n"
                "Normalize each symptom into its most common or textbook form.\n"
                "Count synonyms and paraphrases in the frequency.\n"
                "Return your answer using the 'symptoms_with_frequencies_mentioned' function.\n\n"
                f"SECTION TEXT: {text}"
            ),
        },
    ]

    # Process the query with the LLM tool calls
    time.sleep(1)
    resp = chat_with_retry(input_msg, tools=tools)
    assistant_msg = resp.choices[0].message

    if not assistant_msg.tool_calls:
        print(f"LLM didn't execute the tool call properly for text {text}")
        return {}

    # Return results
    tc = assistant_msg.tool_calls[0]
    function_args = json.loads(tc.function.arguments)
    res = {}
    for item in function_args['symptoms_mentioned']:
        res[item["symptom"]] = item["frequency"]
    
    return res 
    

def extract_query_symptoms(query: str) -> List[str]:
    """
    Takes in a patient query and uses an LLM to extract symptoms.
    Returns the list of symptoms in the query.
    """

    system_instructions = (
        "You are an expert Traditional Chinese Medicine assistant that analyzes patient queries written in Chinese. "
        "Your task is to identify and extract all symptom expressions mentioned by the patient. "
        "You will convert each symptoms into its most commonly known or standard form, as it would appear in a TCM reference text or diagnostic manual.\n\n"
        "INSTRUCTIONS:\n"
        "- Carefully read the patient query and extract every symptom mentioned, even if expressed indirectly (e.g., '後背很緊的感覺' → '背緊').\n"
        "- For each symptom, normalize it into its standard or most widely recognized form.\n"
        "- When a symptom could refer to multiple related symptoms, include all of them (e.g., '頭頸疼' → ['頭痛', '頸痛', '頭頸痛']).\n"
        "- Do not include non-symptom information (e.g., age, ethnicity).\n"
        "- Return your results by calling the provided 'symptoms_mentioned' tool.\n"
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "symptoms_mentioned",
                "description": "Return all TCM symptoms expressed in the patient query, standardized to their common textbook forms.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symptoms": {
                            "type": "array",
                            "description": "List of standardized TCM symptoms expressed in the query.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "symptom": {
                                        "type": "string",
                                        "description": "A symptom in its normalized or commonly known TCM form.",
                                    },
                                },
                                "required": ["symptom"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["symptoms"],
                },
            },
        }
    ]

    # Create the initial message for the query
    input_msgs = [
        {"role": "system", "content": system_instructions},
        {
            "role": "user",
            "content": (
                "Analyze the following Traditional Chinese Medicine patient query and extract all symptoms mentioned.\n"
                "Normalize each symptom into its most common or textbook form.\n"
                "If a symptom could include a combination of symptoms, include all of their common forms as well.\n"
                "Return your answer using the 'symptoms_mentioned' function.\n\n"
                f"QUERY: {query}"
            ),
        },
    ]

    # Process the query with the LLM tool calls
    time.sleep(1)
    resp = chat_with_retry(input_msgs, tools=tools)
    assistant_msg = resp.choices[0].message

    if not assistant_msg.tool_calls:
        print(f"LLM didn't execute the tool call properly for query {query}")
        return []

    # Return results
    tc = assistant_msg.tool_calls[0]
    function_args = json.loads(tc.function.arguments)

    symptoms =[s['symptom'] for s in function_args['symptoms']]

    return symptoms
