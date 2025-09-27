#!/usr/bin/env python3
"""Tag-based expansion system for enhanced TCM knowledge retrieval."""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from dataclasses import dataclass
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TagVector:
    """Represents a tag with its vector embedding."""
    tag_zh: str
    tag_en: str
    vector: List[float]
    chapter_id: str
    section_id: str
    
@dataclass
class TagExpansionResult:
    """Result of tag-based expansion."""
    original_chunks: List[Dict[str, Any]]
    expanded_chunks: List[Dict[str, Any]]
    similar_tags: List[Tuple[str, str, float]]  # (tag_zh, tag_en, similarity)
    expansion_metadata: Dict[str, Any]

class TagExpansionSystem:
    """System for tag-based expansion of retrieval results."""
    
    def __init__(self, openai_client=None):
        """Initialize the tag expansion system.
        
        Args:
            openai_client: OpenAI client for generating embeddings
        """
        self.client = openai_client
        self.tag_vectors: List[TagVector] = []
        self.tag_to_chunks: Dict[str, List[Dict[str, Any]]] = {}
        self.tags_loaded = False
        
    def load_tags_and_vectorize(self, tags_file: str = "../data/tags.json", 
                               cache_file: str = "../data/tag_vectors.pkl") -> bool:
        """Load tags from JSON and create vector embeddings.
        
        Args:
            tags_file: Path to tags JSON file
            cache_file: Path to cache vectorized tags
            
        Returns:
            bool: Success status
        """
        try:
            # Try to load from cache first
            if Path(cache_file).exists():
                logger.info("Loading tag vectors from cache...")
                with open(cache_file, 'rb') as f:
                    self.tag_vectors = pickle.load(f)
                self._build_tag_to_chunks_mapping()
                self.tags_loaded = True
                logger.info(f"Loaded {len(self.tag_vectors)} tag vectors from cache")
                return True
            
            # Load tags from JSON
            with open(tags_file, 'r', encoding='utf-8') as f:
                tags_data = json.load(f)
            
            if not self.client:
                logger.error("OpenAI client required for vectorizing tags")
                return False
            
            logger.info("Vectorizing tags...")
            all_tags = []
            
            # Extract all unique tags
            for section in tags_data.get('sections', []):
                chapter_id = section.get('chapter_id', '')
                section_id = section.get('section_id', '')
                
                for tag in section.get('tags', []):
                    tag_zh = tag.get('tag_zh', '')
                    tag_en = tag.get('tag_en', '')
                    
                    if tag_zh and tag_en:
                        # Create combined text for embedding
                        tag_text = f"{tag_zh} ({tag_en})"
                        all_tags.append({
                            'text': tag_text,
                            'tag_zh': tag_zh,
                            'tag_en': tag_en,
                            'chapter_id': chapter_id,
                            'section_id': section_id
                        })
            
            # Remove duplicates based on tag_zh
            unique_tags = {}
            for tag in all_tags:
                if tag['tag_zh'] not in unique_tags:
                    unique_tags[tag['tag_zh']] = tag
            
            all_tags = list(unique_tags.values())
            logger.info(f"Found {len(all_tags)} unique tags to vectorize")
            
            # Batch vectorize tags
            batch_size = 100
            for i in range(0, len(all_tags), batch_size):
                batch = all_tags[i:i + batch_size]
                texts = [tag['text'] for tag in batch]
                
                try:
                    response = self.client.embeddings.create(
                        input=texts,
                        model="text-embedding-3-large"
                    )
                    
                    for j, embedding in enumerate(response.data):
                        tag_info = batch[j]
                        tag_vector = TagVector(
                            tag_zh=tag_info['tag_zh'],
                            tag_en=tag_info['tag_en'],
                            vector=embedding.embedding,
                            chapter_id=tag_info['chapter_id'],
                            section_id=tag_info['section_id']
                        )
                        self.tag_vectors.append(tag_vector)
                        
                except Exception as e:
                    logger.error(f"Error vectorizing tag batch {i//batch_size + 1}: {e}")
                    continue
            
            # Cache the vectors
            with open(cache_file, 'wb') as f:
                pickle.dump(self.tag_vectors, f)
            
            self._build_tag_to_chunks_mapping()
            self.tags_loaded = True
            logger.info(f"Successfully vectorized and cached {len(self.tag_vectors)} tags")
            return True
            
        except Exception as e:
            logger.error(f"Error loading and vectorizing tags: {e}")
            return False
    
    def _build_tag_to_chunks_mapping(self):
        """Build mapping from tags to chunks that contain them."""
        # This would be populated when chunks are loaded with their tags
        # For now, we'll build it when needed during expansion
        pass
    
    def find_similar_tags(self, query_tags: List[str], top_k: int = 10, 
                         similarity_threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find tags similar to the given query tags.
        
        Args:
            query_tags: List of tag strings to find similarities for
            top_k: Number of similar tags to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of (tag_zh, tag_en, similarity_score) tuples
        """
        if not self.tags_loaded or not self.client:
            return []
        
        try:
            # Vectorize query tags
            query_vectors = []
            for tag in query_tags:
                response = self.client.embeddings.create(
                    input=[tag],
                    model="text-embedding-3-large"
                )
                query_vectors.append(response.data[0].embedding)
            
            # Find similarities
            similar_tags = []
            
            for tag_vector in self.tag_vectors:
                max_similarity = 0
                
                # Compare with all query vectors, take max similarity
                for query_vector in query_vectors:
                    similarity = self._cosine_similarity(query_vector, tag_vector.vector)
                    max_similarity = max(max_similarity, similarity)
                
                if max_similarity >= similarity_threshold:
                    similar_tags.append((tag_vector.tag_zh, tag_vector.tag_en, max_similarity))
            
            # Sort by similarity and return top_k
            similar_tags.sort(key=lambda x: x[2], reverse=True)
            return similar_tags[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar tags: {e}")
            return []
    
    def expand_retrieval_with_tags(self, original_chunks: List[Dict[str, Any]], 
                                  structured_db, max_expansion: int = 3) -> TagExpansionResult:
        """Expand retrieval results using tag-based similarity.
        
        Args:
            original_chunks: Original retrieved chunks
            structured_db: Structured vector database for additional retrieval
            max_expansion: Maximum number of additional chunks to retrieve
            
        Returns:
            TagExpansionResult with expanded chunks and metadata
        """
        if not self.tags_loaded:
            logger.warning("Tags not loaded, returning original chunks")
            return TagExpansionResult(
                original_chunks=original_chunks,
                expanded_chunks=[],
                similar_tags=[],
                expansion_metadata={'error': 'Tags not loaded'}
            )
        
        try:
            # Extract tags from original chunks
            original_tags = set()
            for chunk in original_chunks:
                chunk_tags = chunk.get('associated_tags', [])
                for tag in chunk_tags:
                    if isinstance(tag, dict):
                        original_tags.add(tag.get('tag_zh', ''))
                    else:
                        original_tags.add(str(tag))
            
            if not original_tags:
                logger.info("No tags found in original chunks")
                return TagExpansionResult(
                    original_chunks=original_chunks,
                    expanded_chunks=[],
                    similar_tags=[],
                    expansion_metadata={'message': 'No tags in original chunks'}
                )
            
            # Find similar tags
            similar_tags = self.find_similar_tags(list(original_tags), top_k=15)
            
            if not similar_tags:
                logger.info("No similar tags found")
                return TagExpansionResult(
                    original_chunks=original_chunks,
                    expanded_chunks=[],
                    similar_tags=[],
                    expansion_metadata={'message': 'No similar tags found'}
                )
            
            # Find chunks with similar tags
            expanded_chunks = []
            original_chunk_ids = {id(chunk) for chunk in original_chunks}
            
            # Search for chunks containing similar tags
            for tag_zh, tag_en, similarity in similar_tags:
                if len(expanded_chunks) >= max_expansion:
                    break
                
                # Search database for chunks with this tag
                # This is a simplified approach - in practice, you'd want to 
                # maintain a proper tag-to-chunk index
                tag_query = f"{tag_zh} {tag_en}"
                
                try:
                    # Use the structured database to search for chunks with similar tags
                    search_results = structured_db.search(
                        query=tag_query,
                        search_type="patterns",  # Use patterns search for tag-like content
                        top_k=2
                    )
                    
                    for result in search_results:
                        # Avoid duplicates
                        if id(result) not in original_chunk_ids:
                            result['expansion_source'] = {
                                'similar_tag_zh': tag_zh,
                                'similar_tag_en': tag_en,
                                'similarity_score': similarity
                            }
                            expanded_chunks.append(result)
                            original_chunk_ids.add(id(result))
                            
                            if len(expanded_chunks) >= max_expansion:
                                break
                                
                except Exception as e:
                    logger.warning(f"Error searching for tag '{tag_zh}': {e}")
                    continue
            
            expansion_metadata = {
                'original_tags_count': len(original_tags),
                'similar_tags_found': len(similar_tags),
                'chunks_expanded': len(expanded_chunks),
                'expansion_method': 'tag_similarity'
            }
            
            logger.info(f"Tag expansion: {len(original_chunks)} → {len(original_chunks) + len(expanded_chunks)} chunks")
            
            return TagExpansionResult(
                original_chunks=original_chunks,
                expanded_chunks=expanded_chunks,
                similar_tags=similar_tags,
                expansion_metadata=expansion_metadata
            )
            
        except Exception as e:
            logger.error(f"Error in tag-based expansion: {e}")
            return TagExpansionResult(
                original_chunks=original_chunks,
                expanded_chunks=[],
                similar_tags=[],
                expansion_metadata={'error': str(e)}
            )
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def get_expansion_stats(self) -> Dict[str, Any]:
        """Get statistics about the tag expansion system."""
        return {
            'tags_loaded': self.tags_loaded,
            'total_tag_vectors': len(self.tag_vectors),
            'unique_chapters': len(set(tv.chapter_id for tv in self.tag_vectors)),
            'sample_tags': [(tv.tag_zh, tv.tag_en) for tv in self.tag_vectors[:5]]
        }
