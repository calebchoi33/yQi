"""Refactored document processor for handling various document formats and chunking strategies."""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from text_extractors import TextExtractionManager
from chunking_strategies import ChunkingStrategyFactory, TCMMarkerDetector
from chunking_strategies_semantic import ChunkingStrategyFactory as SemanticChunkingStrategyFactory

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Refactored document processor with modular text extraction and chunking."""
    
    def __init__(self, docs_directory: Optional[str] = None):
        """Initialize document processor.
        
        Args:
            docs_directory: Path to directory containing documents
        """
        if docs_directory is None:
            # Default to docs folder in yQi directory
            models_dir = Path(__file__).parent.absolute()
            yqi_root = models_dir.parent
            self.docs_directory = yqi_root / "docs"
        else:
            self.docs_directory = Path(docs_directory)
        
        # Initialize text extraction manager
        self.text_manager = TextExtractionManager()
        
        # Cache for supported extensions
        self._supported_extensions = None
    
    @property
    def supported_extensions(self) -> set:
        """Get supported file extensions."""
        if self._supported_extensions is None:
            self._supported_extensions = self.text_manager.get_supported_extensions()
        return self._supported_extensions
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from document file.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Extracted text content
        """
        try:
            path_obj = Path(file_path)
            return self.text_manager.extract_text(path_obj)
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def detect_tcm_markers(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Detect TCM markers in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with detected markers
        """
        return TCMMarkerDetector.detect_markers(text)
    
    def process_documents_with_chunking(self, 
                                      chunking_method: str = 'regular', 
                                      chunk_size: int = 500, 
                                      overlap: int = 50) -> List[Dict[str, Any]]:
        """Process all documents with customizable chunking method.
        
        Args:
            chunking_method: 'regular', 'chapter', or 'section'
            chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of document chunks with metadata
        """
        chunks = []
        
        logger.info(f"Processing documents in: {self.docs_directory}")
        logger.info(f"Using chunking method: {chunking_method}")
        
        if not self.docs_directory.exists():
            logger.warning(f"Documents directory not found: {self.docs_directory}")
            return chunks
        
        # Create chunking strategy - try semantic first, then fall back to regular
        try:
            if chunking_method.startswith('semantic_') or chunking_method == 'hybrid':
                strategy = SemanticChunkingStrategyFactory.create_strategy(
                    chunking_method, 
                    chunk_size=chunk_size, 
                    overlap=overlap
                )
            else:
                strategy = ChunkingStrategyFactory.create_strategy(
                    chunking_method, 
                    chunk_size=chunk_size, 
                    overlap=overlap
                )
        except ValueError as e:
            logger.error(f"Invalid chunking method: {e}")
            return chunks
        
        # Process each supported file
        supported_files = self.get_supported_files()
        logger.info(f"Found {len(supported_files)} supported files")
        
        for file_path in supported_files:
            logger.info(f"Processing document: {file_path.name}")
            
            try:
                text = self.extract_text_from_file(str(file_path))
                if not text:
                    logger.warning(f"No text extracted from {file_path.name}")
                    continue
                
                logger.info(f"Extracted {len(text)} characters from {file_path.name}")
                
                # Apply chunking strategy
                text_chunks = strategy.chunk_text(text, str(file_path))
                logger.info(f"Created {len(text_chunks)} chunks from {file_path.name}")
                
                # Process chunks and add metadata
                for i, chunk_data in enumerate(text_chunks):
                    chunk_metadata = {
                        'file_path': str(file_path),
                        'chunk_index': i,
                        'total_chunks': len(text_chunks),
                        'chunking_method': chunking_method,
                        'chunk_type': chunk_data.get('type', 'regular'),
                        'chunk_title': chunk_data.get('title', f'Chunk {i+1}')
                    }
                    
                    # Add all chunk-specific metadata
                    for key, value in chunk_data.items():
                        if key != 'text':  # Don't duplicate text in metadata
                            chunk_metadata[key] = value
                    
                    # Create final chunk with preserved metadata
                    final_chunk = {
                        'text': chunk_data['text'],
                        'source': file_path.name,
                        'chunk_id': f"{file_path.stem}_{i}",
                        'metadata': chunk_metadata
                    }
                    
                    # Also preserve semantic metadata at top level for easy access
                    if 'preserved_complete' in chunk_data:
                        final_chunk['preserved_complete'] = chunk_data['preserved_complete']
                    if 'type' in chunk_data:
                        final_chunk['type'] = chunk_data['type']
                    if 'title' in chunk_data:
                        final_chunk['title'] = chunk_data['title']
                    if 'chapter_title' in chunk_data:
                        final_chunk['chapter_title'] = chunk_data['chapter_title']
                    if 'section_title' in chunk_data:
                        final_chunk['section_title'] = chunk_data['section_title']
                    
                    chunks.append(final_chunk)
                    
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                continue
        
        logger.info(f"Processed {len(chunks)} total chunks from {len(supported_files)} documents")
        return chunks
    
    def process_documents(self, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """Legacy method for backward compatibility."""
        return self.process_documents_with_chunking('regular', chunk_size, overlap)
    
    def get_supported_files(self) -> List[Path]:
        """Get list of supported document file paths."""
        if not self.docs_directory.exists():
            return []
        
        return [
            f for f in self.docs_directory.iterdir()
            if f.is_file() and f.suffix.lower() in self.supported_extensions
        ]
    
    def get_available_documents(self) -> List[str]:
        """Get list of available document filenames."""
        return [f.name for f in self.get_supported_files()]
    
    def get_chunking_methods(self) -> List[str]:
        """Get available chunking methods."""
        return ChunkingStrategyFactory.get_available_methods()
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about available documents."""
        files = self.get_supported_files()
        stats = {
            'total_files': len(files),
            'file_types': {},
            'total_size': 0
        }
        
        for file_path in files:
            ext = file_path.suffix.lower()
            stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
            try:
                stats['total_size'] += file_path.stat().st_size
            except OSError:
                pass
        
        return stats


# Legacy compatibility - keep old method names
class DocumentProcessor(DocumentProcessor):
    """Maintain backward compatibility with old method names."""
    
    def extract_text_from_doc(self, file_path: str) -> str:
        """Legacy method name for backward compatibility."""
        return self.extract_text_from_file(file_path)
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Legacy chunking method for backward compatibility."""
        strategy = ChunkingStrategyFactory.create_strategy('regular', chunk_size=chunk_size, overlap=overlap)
        chunks = strategy.chunk_text(text, "")
        return [chunk['text'] for chunk in chunks]
    
    def chunk_by_chapters(self, text: str, max_chunk_size: int = 2000) -> List[Dict[str, Any]]:
        """Legacy chapter chunking method."""
        strategy = ChunkingStrategyFactory.create_strategy('chapter', chunk_size=max_chunk_size)
        return strategy.chunk_text(text, "")
    
    def chunk_by_sections(self, text: str, max_chunk_size: int = 1500) -> List[Dict[str, Any]]:
        """Legacy section chunking method."""
        strategy = ChunkingStrategyFactory.create_strategy('section', chunk_size=max_chunk_size)
        return strategy.chunk_text(text, "")
