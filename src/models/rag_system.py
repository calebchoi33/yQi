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
                 embedding_model: str = "text-embedding-3-large",
                 language_model: str = "gpt-4o-mini",
                 vector_db_path: str = None,
                 api_key: str = None,
                 use_rag: str = "True"):
        """Initialize RAG system.
        
        Args:
            embedding_model: OpenAI embedding model name
            language_model: OpenAI language model name  
            vector_db_path: Path to save/load vector database
            api_key: OpenAI API key
            use_rag: RAG mode - "True", "False", or "Mock"
        """
        self.embedding_model = embedding_model
        self.language_model = language_model
        self.api_key = api_key
        self.use_rag = use_rag
        
        # Vector database: list of (chunk_text, embedding, metadata) tuples
        self.vector_db = []
        
        # Mock retrieval data for testing
        self.mock_retrieval_data = {}
        
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
    
    def _get_embedding(self, text: str) -> List[float]:
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
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
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
    
    def set_mock_retrieval_data(self, mock_data: Dict[str, List[Dict]]):
        """Set mock retrieval data for testing.
        
        Args:
            mock_data: Dictionary mapping queries to list of mock chunks
        """
        self.mock_retrieval_data = mock_data
    
    def retrieve(self, query: str, top_n: int = 3, include_adjacent: bool = True) -> List[Tuple[str, float, Dict]]:
        """Retrieve most similar chunks to query.
        
        Args:
            query: Search query
            top_n: Number of chunks to return
            include_adjacent: Whether to include adjacent chunks for better context
            
        Returns:
            List of (chunk_text, similarity, metadata) tuples
        """
        # Handle different RAG modes
        if self.use_rag == "False":
            return []
        
        if self.use_rag == "Mock":
            # Return mock data if available
            if query in self.mock_retrieval_data:
                mock_chunks = self.mock_retrieval_data[query]
                return [(chunk['text'], chunk.get('similarity', 0.9), chunk.get('metadata', {})) 
                       for chunk in mock_chunks[:top_n]]
            else:
                logger.warning(f"No mock data found for query: {query}")
                return []
        
        # Regular RAG mode (use_rag == "True")
        if not self.vector_db:
            logger.warning("Vector database is empty")
            return []
        
        if not self.openai_available:
            logger.warning("OpenAI not available for embeddings")
            return []
        
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            if query_embedding is None:
                return []
            
            # Calculate similarities
            similarities = []
            for i, (chunk_text, chunk_embedding, metadata) in enumerate(self.vector_db):
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                similarities.append((chunk_text, similarity, metadata, i))
            
            # Sort by similarity and get top chunks
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_chunks = similarities[:top_n]
            
            # If include_adjacent is True, add adjacent chunks for better context
            if include_adjacent:
                adjacent_chunks = set()
                for chunk_text, similarity, metadata, idx in top_chunks:
                    # Add current chunk
                    adjacent_chunks.add(idx)
                    # Add previous chunk if exists
                    if idx > 0:
                        adjacent_chunks.add(idx - 1)
                    # Add next chunk if exists
                    if idx < len(self.vector_db) - 1:
                        adjacent_chunks.add(idx + 1)
                
                # Get all adjacent chunks with their similarities
                result_chunks = []
                for idx in sorted(adjacent_chunks):
                    chunk_text, chunk_embedding, metadata = self.vector_db[idx]
                    # Calculate similarity for adjacent chunks if not already calculated
                    if idx in [x[3] for x in top_chunks]:
                        # Use existing similarity for top chunks
                        similarity = next(x[1] for x in top_chunks if x[3] == idx)
                    else:
                        # Calculate similarity for adjacent chunks
                        similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                    
                    result_chunks.append((chunk_text, similarity, metadata))
                
                return result_chunks
            else:
                # Return only top chunks without adjacent context
                return [(chunk_text, similarity, metadata) for chunk_text, similarity, metadata, _ in top_chunks]
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
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
            
            # Generate response with function calling for bilingual output
            response = client.chat.completions.create(
                model=self.language_model,
                messages=[
                    {'role': 'system', 'content': instruction_prompt},
                    {'role': 'user', 'content': query}
                ],
                functions=[
                    {
                        "name": "provide_tcm_diagnosis",
                        "description": "Provide TCM diagnosis and treatment in both Chinese and English",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "chinese_response": {
                                    "type": "string",
                                    "description": "Complete TCM diagnosis and prescription in Chinese"
                                },
                                "english_response": {
                                    "type": "string",
                                    "description": "Complete TCM diagnosis and prescription in English, equal in meaning to Chinese response"
                                }
                            },
                            "required": ["chinese_response", "english_response"]
                        }
                    }
                ],
                function_call={"name": "provide_tcm_diagnosis"},
                max_tokens=2000,
                temperature=0.1
            )
        
            # Extract function call result
            if response.choices[0].message.function_call:
                function_args = json.loads(response.choices[0].message.function_call.arguments)
                return {
                    'chinese_response': function_args.get('chinese_response', ''),
                    'english_response': function_args.get('english_response', ''),
                    'combined_response': f"{function_args.get('chinese_response', '')}\n\n---\n\n{function_args.get('english_response', '')}"
                }
            else:
                # Fallback to regular response
                return {
                    'chinese_response': response.choices[0].message.content,
                    'english_response': '',
                    'combined_response': response.choices[0].message.content
                }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {e}"
    
    def _translate_chunks_to_english(self, chunks: List[Tuple[str, float, Dict]]) -> List[Tuple[str, str, float, Dict]]:
        """Translate Chinese chunks to English.
        
        Args:
            chunks: List of (chunk_text, similarity, metadata) tuples
            
        Returns:
            List of (chinese_text, english_text, similarity, metadata) tuples
        """
        if not self.openai_available:
            # Return chunks with empty English translations if OpenAI not available
            return [(chunk_text, "", similarity, metadata) for chunk_text, similarity, metadata in chunks]
        
        translated_chunks = []
        
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key or os.getenv('OPENAI_API_KEY'))
            
            for chunk_text, similarity, metadata in chunks:
                # Translate each chunk
                translation_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a professional translator specializing in Traditional Chinese Medicine (TCM). Translate the following Chinese TCM text to English accurately, preserving all medical terminology and concepts. Maintain the original meaning and technical precision."
                        },
                        {
                            "role": "user", 
                            "content": f"Please translate this Chinese TCM text to English:\n\n{chunk_text}"
                        }
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
                
                english_text = translation_response.choices[0].message.content
                translated_chunks.append((chunk_text, english_text, similarity, metadata))
                
        except Exception as e:
            logger.error(f"Error translating chunks: {e}")
            # Return chunks with empty English translations on error
            return [(chunk_text, "", similarity, metadata) for chunk_text, similarity, metadata in chunks]
        
        return translated_chunks
    
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
        
        # Translate chunks to English
        translated_chunks = self._translate_chunks_to_english(retrieved_chunks)
        
        # Generate response (now returns bilingual response dict)
        response_data = self.generate_response(question, retrieved_chunks, system_prompt)
        
        # Handle both old string format and new dict format
        if isinstance(response_data, dict):
            response_text = response_data.get('combined_response', response_data.get('chinese_response', ''))
            chinese_response = response_data.get('chinese_response', '')
            english_response = response_data.get('english_response', '')
        else:
            response_text = response_data
            chinese_response = response_data
            english_response = ''
        
        return {
            'question': question,
            'response': response_text,
            'chinese_response': chinese_response,
            'english_response': english_response,
            'retrieved_chunks': [
                {
                    'text': chunk_text,
                    'english_text': english_text,
                    'similarity': similarity,
                    'metadata': metadata
                }
                for chunk_text, english_text, similarity, metadata in translated_chunks
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
