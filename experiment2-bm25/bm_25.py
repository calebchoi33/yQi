from typing import List, Dict, Any
import jieba
from numpy import extract
from rank_bm25 import BM25Okapi
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time
import json
import sys
import os

# Add the parent directory to the Python path to import from tagging module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tagging.endpoint import chat_with_retry
from tagging.parse_books import parse_book

QUERIES = [
  "病人，男，20歲，亞裔，幾天前開始覺得不舒服，頭痛，身體發熱，怕吹風，有些流汗。",
  "病人，女，20歲，亞裔，自己覺得好像感冒了，頭頸疼，後背很緊的感覺，怕冷，怕風，沒有流汗。",
  "病人，女，20歲，亞裔，自己覺得好像感冒了，怕冷，身體沉重，說話會喘，有些乾嘔，肚子一些漲感，小便不易出。",
  "病人，女，30歲，亞裔，自己覺得好像感冒了，怕冷，頭痛，身體有些發熱，也感覺身體沉重，好幾天沒有大便了。",
  "病人，男，30歲，白人，小便不順暢，身體微微發熱，一直感到口渴。",
  "病人，男，30歲，白人，自己覺得好像感冒了，身體流汗後依然發熱，胸口下方有些悸動，頭暈，身體微微顫動。",
  "病人，女，35歲，亞裔，感冒好多天了，身體疼痛，全身關節也痛，躺在時很難自己轉動身體，拉肚子，有些小便失禁的情況。",
  "病人，女，40歲，白人，最近小便不順暢，有時候也大便困難，大便乾硬，身體偶然發熱，躺下來時會喘咳不舒服。",
  "病人，女，35歲，亞裔，前幾天感冒，嘔吐，現在覺得肚子很漲。",
  "病人，女，40歲，亞裔，脈搏很細弱，白天想睡覺卻睡不著，感覺心裡很煩，無法躺下來安心睡覺。",
  "病人，女，40歲，亞裔，脈搏很細弱也很沉，白天想睡覺卻睡不著，身體疼痛，手腳冷，關節疼痛。",
  "病人，男，50歲，白人，最近覺得口很渴，胸口疼痛，有氣往上逆的感覺，肚子餓卻不想吃東西，吃了就想吐。",
  "病人，女，50歲，亞裔，手腳非常冷，脈搏非常細微。",
  "病人，女，60歲，亞裔，連續幾天拉肚子，心裡越來越煩躁，按肚子覺得有些脹滿的感覺，肚子卻沒有很硬。"
]
EXPECTED_CHUNKS = [
    (3, 16),
    (3, 35),
    (3, 45),
    (4, 15),
    (4, 31),
    (4, 44),
    (5, 56),
    (6, 64),
    (6, 71),
    (9, 23),
    (9, 25),
    (10, 1),
    (10, 28),
    (10, 52)
]


def cross_encoder_rerank_n(queries: List[str], top_k_per_query: List[Dict[str, Any]], n: int):
    """
    Reranks the topk retrieved docs
    Returns a list of this structure: (reranked_score, (orig_score, ((ch_idx, sec_idx), doc))). im dead
    """
    print(f"\n=== Starting cross_encoder_rerank_n ===")
    print(f"Target rerank count (n): {n}")
    
    # Setup the rerank model
    print("Loading rerank model...")
    model_name = "BAAI/bge-reranker-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    print("Model loaded successfully!")

    # Rerank for each query
    reranked_per_query = []
    for i in range(len(queries)):
        print(f"\nProcessing query {i+1}/{len(queries)}")
        query = queries[i]
        top_k = top_k_per_query[i]
        
        pairs = []
        for _, (_, doc) in top_k:
            pairs.append((query, doc))

        # Run the cross_encoder
        print("Running cross-encoder reranking...")
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            ranked = sorted(zip(scores, top_k), reverse=True)
            reranked_per_query.append(ranked[:n])
        print(f"Reranking completed. Top {n} results selected.")

    print(f"\n=== cross_encoder_rerank_n completed ===")
    return reranked_per_query

def bm25_retrieve_k(queries: List[str], parsed_book: List[Dict[str, Any]], k: int, tokenized_queries: List[List[str]] = []) -> List[List[Any]]:
    """
    Retrieves the top k chunks based on a simple bm25.
    Returns a list of top_k retrieved chunks for each query in queries. it includes the score and the ch_idx, sc_idx
    """
    print(f"\n=== Starting bm25_retrieve_k ===")
    print(f"Target k: {k}")
    
    # Create corpus by flattening parsed_book
    print("Creating corpus from parsed book...")
    corpus = []
    for chapter in parsed_book:
        for section in chapter["sections"]:
            corpus.append(((chapter["chapter_idx"], section["section_idx"]), section["section_text"]))
    print(f"Corpus created with {len(corpus)} sections")

    # Tokenize corpus
    print("Setting up jieba dictionary...")
    jieba.set_dictionary('experiment2-bm25/dict.txt.big')    
    print("Tokenizing corpus...")
    tokenized_corpus = [list(jieba.cut(doc)) for _, doc in corpus]

    # Tokenize queries w/ straightforward method
    if not tokenized_queries:
        print("Tokenizing queries...")
        tokenized_queries = [list(jieba.cut(query)) for query in queries]
    else:
        temp = []
        print(f"Using provided tokenized queries.")
        for query_symptoms in tokenized_queries:
            query_list = []
            for symptom in query_symptoms:
                query_list.extend(list(jieba.cut(symptom)))
            temp.append(query_list)
        tokenized_queries = temp

    # Create BM25 index
    print("Creating BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    # Get top matches
    print("\nRetrieving top-k matches for each query...")
    top_k_per_query = []
    for i, tokenized_query in enumerate(tokenized_queries):
        print(f"\nProcessing query {i+1}/{len(tokenized_queries)}")
        print(f"Query tokens: {tokenized_query[:10]}..." if len(tokenized_query) > 10 else f"Query tokens: {tokenized_query}")
        
        scores = bm25.get_scores(tokenized_query)
        
        ranked = sorted(zip(scores, corpus), reverse=True)
        
        top_k_per_query.append(ranked[:k])

    print(f"\n=== bm25_retrieve_k completed ===")
    return top_k_per_query


SYSTEM_INSTRUCTIONS = (
    "You are an expert Traditional Chinese Medicine (TCM) assistant that analyzes patient queries written in Chinese. "
    "Your task is to identify and extract all symptom expressions mentioned by the patient. "
    "You will convert each symptoms into one or more of its most commonly known or standard forms, as it would appear in a TCM reference text or diagnostic manual.\n\n"
    "INSTRUCTIONS:\n"
    "- Carefully read the patient query and extract every symptom mentioned, even if expressed indirectly (e.g., '後背很緊的感覺' → '背緊').\n"
    "- For each symptom, normalize it into a few of its standard or widely recognized forms, aka synonyms.\n"
    "- When a symptom could refer to multiple related symptoms, include all of them (e.g., '頭頸疼' → ['頭痛', '頸痛', '頭頸痛']).\n"
    "- Do not include non-symptom information (e.g., age, ethnicity), but having a cold or illness is a symptom.\n"
    "- Return your results by calling the provided 'symptoms_mentioned' tool.\n"
)

TOOLS = [
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
                        },
                    },
                },
                "required": ["symptoms"],
            },
        },
    }
]

def extract_query_symptoms(queries: List[str]) -> List[List[str]]:
    """
    Takes in a list of patient queries and extracts symptoms for each query.
    Returns each patient query as a list of symptoms.
    """
    print(f"\n=== Starting extract_query_symptoms ===")
    symptoms_by_query = []

    for i, query in enumerate(queries):
        print(f"\nProcessing query {i+1}/{len(queries)}")
        
        # Create the initial message for the section
        input_msgs = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    "Analyze the following Traditional Chinese Medicine patient query and extract all symptoms mentioned.\n"
                    "Normalize each symptom into its most common or textbook form. If there are any synonyms or other common forms for the symptom, include those as well.\n"
                    "If a symptom could include a combination of symptoms, include all of their common forms as well.\n"
                    "Return your answer using the 'symptoms_mentioned' function.\n\n"
                    f"QUERY: {query}"
                ),
            },
        ]


        # Process the section with the LLM tool calls
        print("Sending request to LLM...")
        resp = chat_with_retry(input_msgs, tools=TOOLS)
        assistant_msg = resp.choices[0].message
        print(f"Received response with {len(assistant_msg.tool_calls)} tool calls")

        # Give the LLM 3 chances to call the tool properly
        retries = 0
        while len(assistant_msg.tool_calls) != 1 and retries < 2:
            print(f"Tool call failed, retrying {retries + 1}/2 times...")
            input_msgs[1]["content"] += (
                "Please call the tool properly. The tag_section tool should be called exactly once."
                "If there are no tags for every category, return an empty list for each category."
            )
            time.sleep(1)
            resp = chat_with_retry(input_msgs, tools=TOOLS)
            assistant_msg = resp.choices[0].message
            print(f"Retry response has {len(assistant_msg.tool_calls)} tool calls")
            retries += 1

        # Skip tool call if LLM still didn't do tool call properly
        if len(assistant_msg.tool_calls) != 1:
            print("WARNING: LLM didn't do tool call properly after 3 tries... :(")
            time.sleep(1)
            continue

        # Execute tool
        print("Executing tool call...")
        tc = assistant_msg.tool_calls[0]
        function_args = json.loads(tc.function.arguments)
        
        symptoms =[s['symptom'] for s in function_args['symptoms']]
        print(f"Extracted {len(symptoms)} symptoms: {symptoms}")
        symptoms_by_query.append(symptoms)
        
        # Pause briefly between sections to reduce request pressure
        time.sleep(1)

    print(f"\n=== extract_query_symptoms completed ===")
    return symptoms_by_query


print("\n" + "="*50)
print("STARTING BM25 EXPERIMENT")
print("="*50)

print("\nStep 1: Parsing book...")
parsed_book = parse_book('books/《人紀傷寒論》.txt')
print(f"Book parsed successfully! Found {len(parsed_book)} chapters")

k=25
# print(f"\nStep 2: Extracting symptoms from {len(QUERIES)} queries...")
# query_symptoms = extract_query_symptoms(QUERIES)
# print("\nExtracted symptoms summary:")
# for i, symptoms in enumerate(query_symptoms):
#     print(f"Query:{QUERIES[i]}")
#     print(f"Symptoms:{symptoms}")

# I didn't want to waste tokens so I just added the retrieved symptoms here:
query_symptoms = [
    ['不適', '頭痛', '發熱', '畏風', '自汗'],
    ['感冒', '頭痛', '頸痛', '頭頸痛', '背緊', '怕冷', '怕風', '無汗'],
    ['感冒', '怕冷', '身重', '氣喘', '乾嘔', '腹脹', '小便不利'],
    ['感冒', '怕冷', '頭痛', '發熱', '身重', '便秘'],
    ['小便不暢', '小便困難', '身熱', '微熱', '口渴'],
    ['感冒', '發熱', '胸悸', '頭暈', '身體顫動', '流汗'],
    ['感冒', '身痛', '全身關節痛', '關節痛', '轉身困難', '腹瀉', '小便失禁'],
    ['小便不暢', '大便困難', '大便乾硬', '發熱', '喘', '咳嗽', '臥則不適'],
    ['感冒', '嘔吐', '腹脹'],
    ['脈細', '脈弱', '白天嗜睡', '失眠', '心煩', '難以安睡'],
    ['脈細', '脈弱', '脈沉', '嗜睡', '失眠', '身痛', '手冷', '足冷', '四肢冷', '關節痛'],
    ['口渴', '胸痛', '氣上逆', '胃脘不適', '食慾不振', '噁心', '欲嘔'],
    ['手冷', '足冷', '四肢發冷', '脈細', '脈微'],
    ['腹瀉', '煩躁', '腹脹', '腹滿'],
]

print("\nStep 3: Running BM25 retrieval...")
scores = bm25_retrieve_k(QUERIES, parsed_book, k, query_symptoms)
print("\nBM25 retrieval completed!")
for i, result_set in enumerate(scores):
    print(f"Query {i}:{QUERIES[i]}")
    print('='*50)
    for j, (score, ((ch_sec_idx_tuple), doc)) in enumerate(result_set):
        print(f"Rank:{j}\n")
        print(f"Score:{score}\n")
        if ch_sec_idx_tuple == EXPECTED_CHUNKS[i]:
            print(f"!!BEST MATCH!!\n")
        print(f"Chapter_idx:{ch_sec_idx_tuple[0]}\n")
        print(f"Section_idx:{ch_sec_idx_tuple[1]}\n")
        print(f"Chunk:{doc}\n")
        print('-'*50)
        


print("\n" + "="*50)
print("EXPERIMENT COMPLETED")
print("="*50)