"""Chunking strategies for document processing."""

import logging
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

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


class ChapterChunkingStrategy(ChunkingStrategy):
    """Chapter-based chunking using #CHAPTER markers."""
    
    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size
        self.regular_chunker = RegularChunkingStrategy(max_chunk_size, 200)
    
    def chunk_text(self, text: str, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Split text into chunks based on #CHAPTER markers."""
        if not text.strip():
            return []
        
        markers = TCMMarkerDetector.detect_markers(text)
        chapters = markers['chapters']
        
        if not chapters:
            # No chapter markers found, fall back to regular chunking
            return self.regular_chunker.chunk_text(text, file_path)
        
        chunks = []
        text_length = len(text)
        
        for i, chapter in enumerate(chapters):
            start_pos = chapter['position']
            end_pos = chapters[i + 1]['position'] if i + 1 < len(chapters) else text_length
            
            chapter_text = text[start_pos:end_pos].strip()
            chapter_title = chapter['title']
            
            # If chapter is too large, split it further
            if len(chapter_text) > self.max_chunk_size:
                # Try to split by sections within the chapter
                chapter_sections = []
                for section in markers['sections']:
                    if start_pos <= section['position'] < end_pos:
                        chapter_sections.append(section)
                
                if chapter_sections:
                    # Split by sections
                    for j, section in enumerate(chapter_sections):
                        section_start = section['position']
                        section_end = (chapter_sections[j + 1]['position'] 
                                     if j + 1 < len(chapter_sections) 
                                     else end_pos)
                        
                        section_text = text[section_start:section_end].strip()
                        if section_text:
                            chunks.append({
                                'text': section_text,
                                'type': 'section',
                                'chapter_title': chapter_title,
                                'section_title': section['title'],
                                'title': f"{chapter_title} - {section['title']}"
                            })
                else:
                    # No sections, split by regular chunking
                    chapter_chunks = self.regular_chunker.chunk_text(chapter_text, file_path)
                    for k, chunk_data in enumerate(chapter_chunks):
                        chunk_data.update({
                            'type': 'chapter_part',
                            'chapter_title': chapter_title,
                            'title': f"{chapter_title} (Part {k + 1})"
                        })
                        chunks.append(chunk_data)
            else:
                # Chapter fits in one chunk
                chunks.append({
                    'text': chapter_text,
                    'type': 'chapter',
                    'chapter_title': chapter_title,
                    'title': chapter_title
                })
        
        return chunks


class SectionChunkingStrategy(ChunkingStrategy):
    """Section-based chunking using #SECTION markers."""
    
    def __init__(self, max_chunk_size: int = 1500):
        self.max_chunk_size = max_chunk_size
        self.chapter_chunker = ChapterChunkingStrategy(max_chunk_size)
        self.regular_chunker = RegularChunkingStrategy(max_chunk_size, 100)
    
    def chunk_text(self, text: str, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Split text into chunks based on #SECTION markers."""
        if not text.strip():
            return []
        
        markers = TCMMarkerDetector.detect_markers(text)
        sections = markers['sections']
        chapters = markers['chapters']
        
        if not sections:
            # No section markers found, fall back to chapter chunking
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
            
            if len(section_text) > self.max_chunk_size:
                # Section is too large, split it further
                section_chunks = self.regular_chunker.chunk_text(section_text, file_path)
                for j, chunk_data in enumerate(section_chunks):
                    chunk_data.update({
                        'type': 'section_part',
                        'chapter_title': chapter_title,
                        'section_title': section_title,
                        'title': f"{chapter_title} - {section_title} (Part {j + 1})"
                    })
                    chunks.append(chunk_data)
            else:
                # Section fits in one chunk
                chunks.append({
                    'text': section_text,
                    'type': 'section',
                    'chapter_title': chapter_title,
                    'section_title': section_title,
                    'title': f"{chapter_title} - {section_title}"
                })
        
        return chunks


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
        else:
            raise ValueError(f"Unknown chunking method: {method}")
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """Get list of available chunking methods."""
        return ['regular', 'chapter', 'section']
