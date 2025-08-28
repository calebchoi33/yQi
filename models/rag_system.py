"""RAG (Retrieval-Augmented Generation) system implementation using OpenAI API."""

import os
import logging
import json
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class RAGSystem:
    """Simple RAG system using OpenAI API for embeddings and generation."""
    
    def __init__(self, 
                 embedding_model: str = "text-embedding-3-small",
                 language_model: str = "gpt-4o-mini",
                 vector_db_path: str = None,
                 api_key: str = None):
        """Initialize RAG system.
        
        Args:
            embedding_model: OpenAI embedding model name
            language_model: OpenAI language model name  
            vector_db_path: Path to save/load vector database
            api_key: OpenAI API key
        """
        self.embedding_model = embedding_model
        self.language_model = language_model
        self.api_key = api_key
        
        # Vector database: list of (chunk_text, embedding, metadata) tuples
        self.vector_db = []
        
        # Set default vector DB path
        if vector_db_path is None:
            self.vector_db_path = Path(__file__).parent / "vector_db.pkl"
        else:
            self.vector_db_path = Path(vector_db_path)
        
        # Check if OpenAI is available
        self.openai_available = self._check_openai()
    
    def _check_openai(self) -> bool:
        """Check if OpenAI is available and API key is valid."""
        try:
            import openai
            
            # Set API key if provided
            if self.api_key:
                openai.api_key = self.api_key
            elif os.getenv('OPENAI_API_KEY'):
                openai.api_key = os.getenv('OPENAI_API_KEY')
            else:
                logger.error("No OpenAI API key provided")
                return False
            
            # Test API connection with a simple embedding request
            try:
                client = openai.OpenAI(api_key=openai.api_key)
                response = client.embeddings.create(
                    model=self.embedding_model,
                    input="test"
                )
                logger.info(f"OpenAI API connection successful with model {self.embedding_model}")
                return True
            except Exception as e:
                logger.error(f"OpenAI API test failed: {e}")
                return False
            
        except ImportError:
            logger.error("OpenAI package not installed. Install with: pip install openai")
            return False
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text using OpenAI API.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        if not self.openai_available:
            logger.error("OpenAI API not available")
            return []
        
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key or os.getenv('OPENAI_API_KEY'))
            response = client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score
        """
        if not a or not b or len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def add_chunk_to_database(self, chunk_text: str, metadata: Dict[str, Any] = None) -> bool:
        """Add a text chunk to the vector database.
        
        Args:
            chunk_text: Text content to add
            metadata: Optional metadata about the chunk
            
        Returns:
            True if successful, False otherwise
        """
        if not chunk_text.strip():
            return False
        
        embedding = self.get_embedding(chunk_text)
        if not embedding:
            return False
        
        self.vector_db.append((chunk_text, embedding, metadata or {}))
        return True
    
    def build_database_from_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Build vector database from document chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and optional metadata
            
        Returns:
            Number of chunks successfully added
        """
        added_count = 0
        
        for i, chunk in enumerate(chunks):
            if 'text' not in chunk:
                logger.warning(f"Chunk {i} missing 'text' field")
                continue
            
            if self.add_chunk_to_database(chunk['text'], chunk.get('metadata', {})):
                added_count += 1
                logger.info(f"Added chunk {added_count}/{len(chunks)} to database")
            else:
                logger.warning(f"Failed to add chunk {i}")
        
        logger.info(f"Successfully added {added_count}/{len(chunks)} chunks to vector database")
        return added_count
    
    def retrieve(self, query: str, top_n: int = 3) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Retrieve most relevant chunks for a query.
        
        Args:
            query: Search query
            top_n: Number of top results to return
            
        Returns:
            List of (chunk_text, similarity_score, metadata) tuples
        """
        if not self.vector_db:
            logger.warning("Vector database is empty")
            return []
        
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        similarities = []
        for chunk_text, embedding, metadata in self.vector_db:
            similarity = self.cosine_similarity(query_embedding, embedding)
            similarities.append((chunk_text, similarity, metadata))
        
        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_n]
    
    def generate_response(self, query: str, retrieved_chunks: List[Tuple[str, float, Dict[str, Any]]], 
                         system_prompt: str = None) -> str:
        """Generate response using retrieved chunks as context.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved context chunks
            system_prompt: Optional custom system prompt
            
        Returns:
            Generated response
        """
        if not self.openai_available:
            return "RAG system not available. OpenAI API not configured."
        
        if not retrieved_chunks:
            return "No relevant context found in the knowledge base."
        
        # Build context from retrieved chunks
        context_parts = []
        for chunk_text, similarity, metadata in retrieved_chunks:
            source = metadata.get('source', 'Unknown')
            context_parts.append(f"From {source}: {chunk_text}")
        
        context = '\n'.join(context_parts)
        
        # Default system prompt for TCM evaluation
        if system_prompt is None:
            system_prompt = """You are an expert Traditional Chinese Medicine (TCM) practitioner with extensive knowledge of TCM diagnosis, treatment principles, and herbal prescriptions. 

Use only the following pieces of context from TCM documents to answer the question. Provide:
1. A detailed TCM diagnostic analysis including symptom analysis and syndrome differentiation (辨證論治)
2. Specific Chinese herbal prescriptions with individual herbs and recommended dosages

Base your response strictly on the provided context. If the context doesn't contain enough information, state that clearly."""
        
        instruction_prompt = f"""{system_prompt}

Context from TCM documents:
{context}

Please answer based only on the above context."""
        
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key or os.getenv('OPENAI_API_KEY'))
            
            # Generate response
            response = client.chat.completions.create(
                model=self.language_model,
                messages=[
                    {'role': 'system', 'content': instruction_prompt},
                    {'role': 'user', 'content': query}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {e}"
    
    def query(self, question: str, top_n: int = 3, system_prompt: str = None) -> Dict[str, Any]:
        """Complete RAG query: retrieve and generate.
        
        Args:
            question: User question
            top_n: Number of chunks to retrieve
            system_prompt: Optional custom system prompt
            
        Returns:
            Dictionary with response, retrieved chunks, and metadata
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(question, top_n)
        
        # Generate response
        response = self.generate_response(question, retrieved_chunks, system_prompt)
        
        return {
            'question': question,
            'response': response,
            'retrieved_chunks': [
                {
                    'text': chunk_text,
                    'similarity': similarity,
                    'metadata': metadata
                }
                for chunk_text, similarity, metadata in retrieved_chunks
            ],
            'num_chunks_used': len(retrieved_chunks)
        }
    
    def save_database(self, path: str = None) -> bool:
        """Save vector database to file.
        
        Args:
            path: Optional custom path
            
        Returns:
            True if successful
        """
        save_path = Path(path) if path else self.vector_db_path
        
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(self.vector_db, f)
            logger.info(f"Vector database saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving database: {e}")
            return False
    
    def load_database(self, path: str = None) -> bool:
        """Load vector database from file.
        
        Args:
            path: Optional custom path
            
        Returns:
            True if successful
        """
        load_path = Path(path) if path else self.vector_db_path
        
        if not load_path.exists():
            logger.info(f"No existing database found at {load_path}")
            return False
        
        try:
            with open(load_path, 'rb') as f:
                self.vector_db = pickle.load(f)
            logger.info(f"Vector database loaded from {load_path} ({len(self.vector_db)} chunks)")
            return True
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the current database.
        
        Returns:
            Database statistics
        """
        if not self.vector_db:
            return {'num_chunks': 0, 'sources': []}
        
        sources = set()
        for _, _, metadata in self.vector_db:
            if 'source' in metadata:
                sources.add(metadata['source'])
        
        return {
            'num_chunks': len(self.vector_db),
            'sources': list(sources),
            'openai_available': self.openai_available,
            'embedding_model': self.embedding_model,
            'language_model': self.language_model
        }
