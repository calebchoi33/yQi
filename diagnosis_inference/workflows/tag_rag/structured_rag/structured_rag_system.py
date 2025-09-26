#!/usr/bin/env python3
"""Enhanced RAG System using Structured Vector Database with multi-vector search."""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import os

from structured_vector_db import StructuredVectorDB, VectorRecord

# Try to import OpenAI
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class StructuredRAGSystem:
    """Enhanced RAG system using structured vector database with multi-vector search."""
    
    def __init__(self, config_path: str, db_path: str, api_key: str = None):
        """Initialize the structured RAG system.
        
        Args:
            config_path: Path to structured vector database configuration
            db_path: Path to the structured vector database file
            api_key: OpenAI API key
        """
        self.config_path = config_path
        self.db_path = db_path
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        # Initialize OpenAI client
        self.openai_available = False
        if OPENAI_AVAILABLE and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                self.openai_available = True
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
        # Initialize structured vector database
        self.structured_db = StructuredVectorDB(config_path, self.client if self.openai_available else None)
        self.database_loaded = False
        
        # Load tags data
        self.tags_data = self._load_tags()
        
        # Initialize tag expansion system
        from tag_expansion_system import TagExpansionSystem
        self.tag_expansion = TagExpansionSystem(self.client if self.openai_available else None)
        
        logger.info("Initialized StructuredRAGSystem")
    
    def _load_tags(self) -> Dict[str, Any]:
        """Load tags data from tags.json file."""
        tags_file = "../data/tags.json"
        try:
            with open(tags_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading tags: {e}")
            return {}
    
    def _get_tags_for_chunk(self, chunk: Dict[str, Any]) -> List[str]:
        """Get associated tags for a chunk based on chapter and section."""
        if not self.tags_data or 'sections' not in self.tags_data:
            return []
        
        chapter_index = chunk.get('metadata', {}).get('chapter_index', 0)
        section_index = chunk.get('metadata', {}).get('section_index', 0)
        
        # Find matching section in tags data
        for section in self.tags_data['sections']:
            if (str(section.get('chapter_id', '')) == str(chapter_index) and 
                str(section.get('section_id', '')) == str(section_index)):
                return section.get('tags', [])
        
        return []
    
    def load_database(self) -> bool:
        """Load the structured vector database."""
        if not Path(self.db_path).exists():
            logger.warning(f"Database file not found: {self.db_path}")
            return False
        
        success = self.structured_db.load_database(self.db_path)
        self.database_loaded = success
        
        # Load and vectorize tags for expansion
        if success and self.openai_available:
            self.tag_expansion.load_tags_and_vectorize()
        
        return success
    
    def build_database(self) -> bool:
        """Build the structured vector database from configured documents."""
        if not self.openai_available:
            logger.error("Cannot build database without OpenAI API access")
            return False
        
        total_records = self.structured_db.build_from_documents()
        if total_records > 0:
            # Save the database
            self.structured_db.save_database(self.db_path)
            self.database_loaded = True
            return True
        
        return False
    
    def search(self, query: str, search_type: str = "multi_vector", top_k: int = 5,
               vector_weights: Optional[Dict[str, float]] = None, 
               expand_adjacent: bool = True, use_tag_expansion: bool = True) -> List[Dict[str, Any]]:
        """Search the knowledge base using various methods.
        
        Args:
            query: Search query
            search_type: Type of search ("multi_vector", "symptoms", "organs", "formulas", "patterns")
            top_k: Number of top results to return
            vector_weights: Custom weights for multi-vector search
            expand_adjacent: Whether to include adjacent content
            use_tag_expansion: Whether to expand query with TCM tags
            
        Returns:
            List of search results with content and metadata
        """
        if not self.database_loaded:
            logger.error("Database not loaded")
            return []
        
        # Perform search based on type
        if search_type == "multi_vector":
            if use_tag_expansion:
                results = self.structured_db.search_with_tags(query, vector_weights, top_k, use_tag_expansion)
            else:
                results = self.structured_db.search_multi_vector(query, vector_weights, top_k)
        else:
            # Single vector search with optional tag expansion
            single_weights = {f"{search_type}_vector": 1.0}
            if use_tag_expansion:
                results = self.structured_db.search_with_tags(query, single_weights, top_k, use_tag_expansion)
            else:
                results = self.structured_db.search_multi_vector(query, single_weights, top_k)
        
        # Process results
        processed_results = []
        for record, score in results:
            result_data = {
                "content": record.full_content,
                "score": score,
                "metadata": {
                    "book_name": record.book_name,
                    "book_id": record.book_id,
                    "chapter_index": record.chapter_index,
                    "chapter_title": record.chapter_title,
                    "section_index": record.section_index,
                    "section_title": record.section_title,
                    "content_type": record.content_type,
                    "word_count": record.word_count,
                    "char_count": record.char_count
                },
                "specialized_content": {
                    "symptoms": record.symptoms_content,
                    "organs": record.organs_content,
                    "formulas": record.formulas_content,
                    "patterns": record.patterns_content
                }
            }
            
            # Add adjacent content if requested
            if expand_adjacent:
                adjacent_records = self.structured_db.get_adjacent_content(
                    record, 
                    expand_sections=2,
                    expand_to_chapter=False
                )
                result_data["adjacent_content"] = [
                    {
                        "content": adj_record.full_content,
                        "chapter_title": adj_record.chapter_title,
                        "section_title": adj_record.section_title,
                        "section_index": adj_record.section_index
                    }
                    for adj_record in adjacent_records if adj_record != record
                ]
            
            processed_results.append(result_data)
        
        return processed_results
    
    def generate_response(self, query: str, search_results: List[Dict], 
                         system_prompt: str = None) -> Dict[str, Any]:
        """Generate a response using search results and LLM.
        
        Args:
            query: User query
            search_results: Results from knowledge search
            system_prompt: Custom system prompt
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.openai_available:
            return {
                "error": "OpenAI API not available",
                "chinese_response": "",
                "english_response": "",
                "combined_response": ""
            }
        
        # Default system prompt for TCM
        if not system_prompt:
            system_prompt = """You are an expert Traditional Chinese Medicine (TCM) practitioner with extensive knowledge of TCM diagnosis, treatment principles, and herbal prescriptions. 

Use the provided context from TCM documents to answer questions. You must provide your response in BOTH Chinese and English.

Provide:
1) A detailed TCM diagnostic analysis including symptom analysis and syndrome differentiation (辨證論治)
2) Specific Chinese herbal prescriptions with individual herbs and recommended dosages
3) Treatment principles and lifestyle recommendations

Base your response strictly on the provided context. Always respond with Chinese first, then English."""
        
        # Prepare context from search results
        context_parts = []
        for i, result in enumerate(search_results[:3]):  # Use top 3 results
            context_parts.append(f"Context {i+1} (from {result['metadata']['book_name']}, Chapter {result['metadata']['chapter_index']}):")
            context_parts.append(result['content'])
            
            # Add specialized content if available
            specialized = result.get('specialized_content', {})
            for content_type, content in specialized.items():
                if content and content.strip():
                    context_parts.append(f"{content_type.title()}: {content}")
            
            context_parts.append("")  # Add spacing
        
        context = "\n".join(context_parts)
        
        # Define function for structured bilingual response
        function_definition = {
            "name": "provide_tcm_diagnosis",
            "description": "Provide TCM diagnosis and treatment in both Chinese and English",
            "parameters": {
                "type": "object",
                "properties": {
                    "chinese_response": {
                        "type": "string",
                        "description": "Complete TCM diagnosis and treatment response in Chinese"
                    },
                    "english_response": {
                        "type": "string", 
                        "description": "Complete TCM diagnosis and treatment response in English"
                    }
                },
                "required": ["chinese_response", "english_response"]
            }
        }
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ],
                functions=[function_definition],
                function_call={"name": "provide_tcm_diagnosis"},
                temperature=0.3
            )
            
            if response.choices[0].message.function_call:
                function_args = json.loads(response.choices[0].message.function_call.arguments)
                chinese_response = function_args.get("chinese_response", "")
                english_response = function_args.get("english_response", "")
                
                # Add tags to each retrieved chunk
                enriched_chunks = []
                for chunk in search_results:
                    enriched_chunk = chunk.copy()
                    enriched_chunk['associated_tags'] = self._get_tags_for_chunk(chunk)
                    enriched_chunks.append(enriched_chunk)
                
                return {
                    "chinese_response": chinese_response,
                    "english_response": english_response,
                    "combined_response": f"{chinese_response}\n\n---\n\n{english_response}",
                    "search_results_used": len(search_results),
                    "retrieved_chunks": enriched_chunks,
                    "model": "gpt-4o-mini"
                }
            else:
                # Fallback to regular response
                content = response.choices[0].message.content
                # Add tags to each retrieved chunk
                enriched_chunks = []
                for chunk in search_results:
                    enriched_chunk = chunk.copy()
                    enriched_chunk['associated_tags'] = self._get_tags_for_chunk(chunk)
                    enriched_chunks.append(enriched_chunk)
                
                return {
                    "chinese_response": content,
                    "english_response": "",
                    "combined_response": content,
                    "search_results_used": len(search_results),
                    "retrieved_chunks": enriched_chunks,
                    "model": "gpt-4o-mini"
                }
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "error": str(e),
                "chinese_response": "",
                "english_response": "",
                "combined_response": ""
            }
    
    def query(self, query: str, search_type: str = "multi_vector", 
              use_tag_expansion: bool = True, top_k: int = 5, 
              enable_tag_expansion: bool = True) -> Dict[str, Any]:
        """Complete query pipeline: search + generate response.
        
        Args:
            query: User query
            search_type: Type of search to perform
            vector_weights: Custom vector weights for multi-vector search
            system_prompt: Custom system prompt
            use_tag_expansion: Whether to expand query with TCM tags
            
        Returns:
            Complete response with search results and generated answer
        """
        # Search knowledge base
        search_results = self.search(
            query=query,
            search_type=search_type,
            top_k=top_k,
            expand_adjacent=True,
            use_tag_expansion=use_tag_expansion
        )
        
        # Apply tag-based expansion if enabled
        if enable_tag_expansion and search_results:
            expansion_result = self.tag_expansion.expand_retrieval_with_tags(
                original_chunks=search_results,
                structured_db=self.structured_db,
                max_expansion=3
            )
            
            # Combine original and expanded chunks
            all_chunks = expansion_result.original_chunks + expansion_result.expanded_chunks
            search_results = all_chunks[:top_k + 3]  # Allow slightly more for expansion
        
        if not search_results:
            return {
                "error": "No relevant knowledge found",
                "chinese_response": "未找到相关知识",
                "english_response": "No relevant knowledge found",
                "combined_response": "未找到相关知识\n\nNo relevant knowledge found",
                "search_results": []
            }
        
        # Generate response
        response_data = self.generate_response(query, search_results)
        
        # Add search results to response
        response_data["search_results"] = search_results
        response_data["search_type"] = search_type
        
        return response_data
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the loaded database."""
        if not self.database_loaded:
            return {"error": "Database not loaded"}
        
        return self.structured_db.get_database_info()
    
    def get_available_search_types(self) -> List[str]:
        """Get available search types based on vector columns."""
        if not self.database_loaded:
            return []
        
        vector_columns = list(self.structured_db.vector_columns.keys())
        search_types = ["multi_vector"]
        
        # Add individual vector types (remove '_vector' suffix)
        for col in vector_columns:
            if col.endswith('_vector'):
                search_types.append(col.replace('_vector', ''))
        
        return search_types
