Semantic BM25 workflow

- Per-term ingestion of all tag families from `tagging/output/《人紀傷寒論》_tags.json`.
- Embeds each individual tag term (Chinese and English) with OpenAI `text-embedding-3-small` (1536-dim, normalized).
- Stores one row per term in `terms` vec0 table and a `sections` table with per-section length.
- Query scores all sections using the provided semantic BM25 formulation with k1,b in the TF denominator, no query tokenization.

Files
- embeddings.py: term extraction and embeddings
- database.py: sqlite-vec schema and helpers
- ingestion.py: builds DB `semantic_bm25_vec.db`
- query_engine.py: `score(query_text, top_k=5)`

Usage
1) export OPENAI_API_KEY and install requirements
2) run `python ingestion.py`
3) run `python query_engine.py` or import and call `score()`
