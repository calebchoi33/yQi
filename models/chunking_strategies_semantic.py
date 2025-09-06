"""Enhanced chunking strategies that preserve semantic boundaries."""

import logging
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import tiktoken

logger = logging.getLogger(__name__)


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk_text(self, text: str, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Chunk text according to the strategy."""
        pass


class RegularChunkingStrategy(ChunkingStrategy):
    """Regular fixed-size chunking with overlap."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + self.chunk_size - 100:
                    end = sentence_end + 1
                else:
                    # Look for paragraph breaks
                    para_break = text.rfind('\n\n', start, end)
                    if para_break > start + self.chunk_size - 200:
                        end = para_break + 2
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({
                    'text': chunk,
                    'type': 'regular',
                    'title': f'Chunk {len(chunks) + 1}'
                })
            
            start = end - self.overlap
        
        return chunks


class TCMMarkerDetector:
    """Utility class for detecting TCM markers in text."""
    
    @staticmethod
    def detect_markers(text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Detect TCM markers (#CHAPTER, #SECTION, #FORMULA) and their positions."""
        markers = {
            'chapters': [],
            'sections': [],
            'formulas': []
        }
        
        lines = text.split('\n')
        char_position = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            if line_stripped == '#CHAPTER':
                markers['chapters'].append({
                    'position': char_position,
                    'line_number': i + 1,
                    'title': lines[i + 1].strip() if i + 1 < len(lines) else 'Unknown Chapter'
                })
            elif line_stripped == '#SECTION':
                markers['sections'].append({
                    'position': char_position,
                    'line_number': i + 1,
                    'title': lines[i + 1].strip() if i + 1 < len(lines) else 'Unknown Section'
                })
            elif line_stripped == '#FORMULA':
                markers['formulas'].append({
                    'position': char_position,
                    'line_number': i + 1,
                    'title': lines[i + 1].strip() if i + 1 < len(lines) else 'Unknown Formula'
                })
            
            char_position += len(line) + 1  # +1 for newline character
        
        return markers


class SemanticChapterChunkingStrategy(ChunkingStrategy):
    """Chapter-based chunking that preserves complete chapters with token limits."""
    
    def __init__(self, fallback_chunk_size: int = 2000, max_tokens: int = 7500):
        """
        Args:
            fallback_chunk_size: Only used when no chapter markers exist
            max_tokens: Maximum tokens per chunk (safety margin below 8191)
        """
        self.fallback_chunk_size = fallback_chunk_size
        self.max_tokens = max_tokens
        self.regular_chunker = RegularChunkingStrategy(fallback_chunk_size, 200)
        self.encoding = tiktoken.get_encoding('cl100k_base')
    
    def chunk_text(self, text: str, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Split text into complete chapters, preserving semantic boundaries."""
        if not text.strip():
            return []
        
        markers = TCMMarkerDetector.detect_markers(text)
        chapters = markers['chapters']
        
        if not chapters:
            # No chapter markers found, fall back to regular chunking
            logger.info(f"No chapter markers found in {file_path}, using regular chunking")
            return self.regular_chunker.chunk_text(text, file_path)
        
        chunks = []
        text_length = len(text)
        
        for i, chapter in enumerate(chapters):
            start_pos = chapter['position']
            end_pos = chapters[i + 1]['position'] if i + 1 < len(chapters) else text_length
            
            chapter_text = text[start_pos:end_pos].strip()
            chapter_title = chapter['title']
            
            if chapter_text:
                # Check token count and split if necessary
                token_count = len(self.encoding.encode(chapter_text))
                
                if token_count <= self.max_tokens:
                    # Chapter fits within token limit
                    chunks.append({
                        'text': chapter_text,
                        'type': 'semantic_chapter',
                        'chapter_title': chapter_title,
                        'title': chapter_title,
                        'char_count': len(chapter_text),
                        'token_count': token_count,
                        'preserved_complete': True
                    })
                    logger.debug(f"Created semantic chapter chunk: {chapter_title} ({len(chapter_text)} chars, {token_count} tokens)")
                else:
                    # Chapter too large, split by sections or regular chunking
                    logger.warning(f"Chapter '{chapter_title}' exceeds token limit ({token_count} > {self.max_tokens}), splitting...")
                    
                    # Try to find sections within this chapter
                    markers = TCMMarkerDetector.detect_markers(chapter_text)
                    chapter_sections = [s for s in markers['sections'] if start_pos <= s['position'] < end_pos]
                    
                    if chapter_sections:
                        # Split by sections
                        for j, section in enumerate(chapter_sections):
                            section_start = section['position'] - start_pos  # Relative to chapter start
                            section_end = (chapter_sections[j + 1]['position'] - start_pos 
                                         if j + 1 < len(chapter_sections) 
                                         else len(chapter_text))
                            
                            section_text = chapter_text[section_start:section_end].strip()
                            section_tokens = len(self.encoding.encode(section_text))
                            
                            if section_tokens <= self.max_tokens:
                                chunks.append({
                                    'text': section_text,
                                    'type': 'chapter_section',
                                    'chapter_title': chapter_title,
                                    'section_title': section['title'],
                                    'title': f"{chapter_title} - {section['title']}",
                                    'char_count': len(section_text),
                                    'token_count': section_tokens,
                                    'preserved_complete': True
                                })
                            else:
                                # Section still too large, use regular chunking
                                section_chunks = self.regular_chunker.chunk_text(section_text, file_path)
                                for k, chunk_data in enumerate(section_chunks):
                                    chunk_data.update({
                                        'type': 'chapter_section_part',
                                        'chapter_title': chapter_title,
                                        'section_title': section['title'],
                                        'title': f"{chapter_title} - {section['title']} (Part {k + 1})",
                                        'preserved_complete': False
                                    })
                                    chunks.append(chunk_data)
                    else:
                        # No sections, use regular chunking on entire chapter
                        chapter_chunks = self.regular_chunker.chunk_text(chapter_text, file_path)
                        for k, chunk_data in enumerate(chapter_chunks):
                            chunk_data.update({
                                'type': 'chapter_part',
                                'chapter_title': chapter_title,
                                'title': f"{chapter_title} (Part {k + 1})",
                                'preserved_complete': False
                            })
                            chunks.append(chunk_data)
        
        return chunks


class SemanticSectionChunkingStrategy(ChunkingStrategy):
    """Section-based chunking that preserves complete sections with token limits."""
    
    def __init__(self, fallback_chunk_size: int = 1500, max_tokens: int = 7500):
        """
        Args:
            fallback_chunk_size: Only used when no section markers exist
            max_tokens: Maximum tokens per chunk (safety margin below 8191)
        """
        self.fallback_chunk_size = fallback_chunk_size
        self.max_tokens = max_tokens
        self.chapter_chunker = SemanticChapterChunkingStrategy(fallback_chunk_size, max_tokens)
        self.regular_chunker = RegularChunkingStrategy(fallback_chunk_size, 100)
        self.encoding = tiktoken.get_encoding('cl100k_base')
    
    def chunk_text(self, text: str, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Split text into complete sections, preserving semantic boundaries."""
        if not text.strip():
            return []
        
        markers = TCMMarkerDetector.detect_markers(text)
        sections = markers['sections']
        chapters = markers['chapters']
        
        if not sections:
            # No section markers found, fall back to chapter chunking
            logger.info(f"No section markers found in {file_path}, using chapter chunking")
            return self.chapter_chunker.chunk_text(text, file_path)
        
        chunks = []
        text_length = len(text)
        
        # Create a mapping of sections to their parent chapters
        section_to_chapter = {}
        for section in sections:
            parent_chapter = None
            for chapter in chapters:
                if chapter['position'] <= section['position']:
                    parent_chapter = chapter['title']
                else:
                    break
            section_to_chapter[section['position']] = parent_chapter or 'Unknown Chapter'
        
        for i, section in enumerate(sections):
            start_pos = section['position']
            end_pos = sections[i + 1]['position'] if i + 1 < len(sections) else text_length
            
            section_text = text[start_pos:end_pos].strip()
            section_title = section['title']
            chapter_title = section_to_chapter[section['position']]
            
            if section_text:
                # Check token count and split if necessary
                token_count = len(self.encoding.encode(section_text))
                
                if token_count <= self.max_tokens:
                    # Section fits within token limit
                    chunks.append({
                        'text': section_text,
                        'type': 'semantic_section',
                        'chapter_title': chapter_title,
                        'section_title': section_title,
                        'title': f"{chapter_title} - {section_title}",
                        'char_count': len(section_text),
                        'token_count': token_count,
                        'preserved_complete': True
                    })
                    logger.debug(f"Created semantic section chunk: {section_title} ({len(section_text)} chars, {token_count} tokens)")
                else:
                    # Section too large, split with regular chunking
                    logger.warning(f"Section '{section_title}' exceeds token limit ({token_count} > {self.max_tokens}), splitting...")
                    section_chunks = self.regular_chunker.chunk_text(section_text, file_path)
                    for j, chunk_data in enumerate(section_chunks):
                        chunk_data.update({
                            'type': 'section_part',
                            'chapter_title': chapter_title,
                            'section_title': section_title,
                            'title': f"{chapter_title} - {section_title} (Part {j + 1})",
                            'preserved_complete': False
                        })
                        chunks.append(chunk_data)
        
        return chunks


class HybridChunkingStrategy(ChunkingStrategy):
    """Hybrid strategy: semantic for marked content, regular for unmarked content."""
    
    def __init__(self, semantic_method: str = 'section', regular_chunk_size: int = 1000, max_tokens: int = 7500):
        self.semantic_method = semantic_method
        self.regular_chunk_size = regular_chunk_size
        self.max_tokens = max_tokens
        
        if semantic_method == 'chapter':
            self.semantic_chunker = SemanticChapterChunkingStrategy(regular_chunk_size, max_tokens)
        else:
            self.semantic_chunker = SemanticSectionChunkingStrategy(regular_chunk_size, max_tokens)
        
        self.regular_chunker = RegularChunkingStrategy(regular_chunk_size, 100)
    
    def chunk_text(self, text: str, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Use semantic chunking for marked content, regular for unmarked."""
        if not text.strip():
            return []
        
        markers = TCMMarkerDetector.detect_markers(text)
        has_markers = bool(markers['chapters'] or markers['sections'])
        
        if has_markers:
            logger.info(f"Using semantic {self.semantic_method} chunking for {file_path}")
            return self.semantic_chunker.chunk_text(text, file_path)
        else:
            logger.info(f"No TCM markers found, using regular chunking for {file_path}")
            return self.regular_chunker.chunk_text(text, file_path)


class ChunkingStrategyFactory:
    """Factory for creating chunking strategies."""
    
    @staticmethod
    def create_strategy(method: str, **kwargs) -> ChunkingStrategy:
        """Create a chunking strategy based on method name."""
        if method == 'regular':
            return RegularChunkingStrategy(
                chunk_size=kwargs.get('chunk_size', 500),
                overlap=kwargs.get('overlap', 50)
            )
        elif method == 'chapter':
            return ChapterChunkingStrategy(
                max_chunk_size=kwargs.get('chunk_size', 2000)
            )
        elif method == 'section':
            return SectionChunkingStrategy(
                max_chunk_size=kwargs.get('chunk_size', 1500)
            )
        elif method == 'semantic_chapter':
            return SemanticChapterChunkingStrategy(
                fallback_chunk_size=kwargs.get('chunk_size', 2000),
                max_tokens=kwargs.get('max_tokens', 7500)
            )
        elif method == 'semantic_section':
            return SemanticSectionChunkingStrategy(
                fallback_chunk_size=kwargs.get('chunk_size', 1500),
                max_tokens=kwargs.get('max_tokens', 7500)
            )
        elif method == 'hybrid':
            return HybridChunkingStrategy(
                semantic_method=kwargs.get('semantic_method', 'section'),
                regular_chunk_size=kwargs.get('chunk_size', 1000),
                max_tokens=kwargs.get('max_tokens', 7500)
            )
        else:
            raise ValueError(f"Unknown chunking method: {method}")
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """Get list of available chunking methods."""
        return ['regular', 'chapter', 'section', 'semantic_chapter', 'semantic_section', 'hybrid']


# Backward compatibility - import the old constrained strategies
from chunking_strategies import (
    ChapterChunkingStrategy, 
    SectionChunkingStrategy
)
