#!/usr/bin/env python3
"""Create new vector database with configurable chunking options."""

import json
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

# Add models directory to path
sys.path.append(str(Path(__file__).parent.parent / 'models'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(str(Path(__file__).parent.parent / 'evaluation' / '.env'))

from rag_system import RAGSystem
from document_processor import DocumentProcessor

def load_vdb_config(config_path: str) -> Dict[str, Any]:
    """Load vector database configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

class AdvancedDocumentProcessor(DocumentProcessor):
    """Enhanced document processor with chapter-aware chunking."""
    
    def __init__(self, docs_directory: str, config: Dict[str, Any]):
        super().__init__(docs_directory)
        self.config = config
        self.chunk_method = config.get('chunking_method', 'paragraph')
        self.chunk_size = config.get('chunk_size', 500)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        self.embed_chapter_info = config.get('embed_chapter_info', True)
    
    def detect_chapters(self, text: str) -> List[Dict[str, Any]]:
        """Detect chapter boundaries in TCM texts."""
        chapters = []
        
        # Common TCM chapter patterns
        chapter_patterns = [
            r'第[一二三四五六七八九十百千]+章',  # 第一章, 第二章, etc.
            r'第[0-9]+章',  # 第1章, 第2章, etc.
            r'《[^》]+》',  # 《金匱要略》, 《傷寒論》, etc.
            r'[一二三四五六七八九十百千]+、',  # 一、二、三、etc.
            r'[0-9]+\.',  # 1. 2. 3. etc.
            r'^[^\n]{1,50}$(?=\n\n)',  # Short lines followed by double newline (likely titles)
        ]
        
        combined_pattern = '|'.join(f'({pattern})' for pattern in chapter_patterns)
        
        for match in re.finditer(combined_pattern, text, re.MULTILINE):
            chapters.append({
                'title': match.group().strip(),
                'start_pos': match.start(),
                'pattern_type': 'chapter_header'
            })
        
        # Sort by position
        chapters.sort(key=lambda x: x['start_pos'])
        
        # Add end positions
        for i, chapter in enumerate(chapters):
            if i < len(chapters) - 1:
                chapter['end_pos'] = chapters[i + 1]['start_pos']
            else:
                chapter['end_pos'] = len(text)
        
        return chapters
    
    def chunk_by_chapter(self, text: str, file_path: str) -> List[Dict[str, Any]]:
        """Chunk text by detected chapters."""
        chapters = self.detect_chapters(text)
        chunks = []
        
        if not chapters:
            # Fallback to paragraph chunking if no chapters detected
            return self.chunk_by_paragraph(text, file_path)
        
        for i, chapter in enumerate(chapters):
            chapter_text = text[chapter['start_pos']:chapter['end_pos']].strip()
            
            if len(chapter_text) <= self.chunk_size:
                # Small chapter, keep as single chunk
                chunk_text = chapter_text
                if self.embed_chapter_info:
                    chunk_text = f"Chapter: {chapter['title']}\n\n{chunk_text}"
                
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'file_path': str(file_path),
                        'chunk_index': len(chunks),
                        'chapter_title': chapter['title'],
                        'chapter_number': i + 1,
                        'chunking_method': 'chapter'
                    }
                })
            else:
                # Large chapter, split into smaller chunks
                chapter_chunks = self.chunk_text(chapter_text, self.chunk_size, self.chunk_overlap)
                
                for j, chunk_text in enumerate(chapter_chunks):
                    if self.embed_chapter_info:
                        chunk_text = f"Chapter: {chapter['title']}\n\n{chunk_text}"
                    
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            'file_path': str(file_path),
                            'chunk_index': len(chunks),
                            'chapter_title': chapter['title'],
                            'chapter_number': i + 1,
                            'chapter_chunk': j + 1,
                            'chunking_method': 'chapter_subdivided'
                        }
                    })
        
        return chunks
    
    def chunk_by_paragraph(self, text: str, file_path: str) -> List[Dict[str, Any]]:
        """Chunk text by paragraphs."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = ""
        
        for i, paragraph in enumerate(paragraphs):
            if len(current_chunk) + len(paragraph) + 2 <= self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': {
                            'file_path': str(file_path),
                            'chunk_index': len(chunks),
                            'chunking_method': 'paragraph'
                        }
                    })
                current_chunk = paragraph + "\n\n"
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    'file_path': str(file_path),
                    'chunk_index': len(chunks),
                    'chunking_method': 'paragraph'
                }
            })
        
        return chunks
    
    def chunk_by_custom(self, text: str, file_path: str) -> List[Dict[str, Any]]:
        """Custom chunking with overlap."""
        return super().chunk_text(text, self.chunk_size, self.chunk_overlap)
    
    def get_supported_files(self) -> List[Path]:
        """Get list of supported document files."""
        supported_extensions = {'.txt', '.docx', '.doc', '.pdf'}
        files = []
        
        for file_path in self.docs_directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append(file_path)
        
        return files
    
    def process_documents(self) -> List[Dict[str, Any]]:
        """Process documents with advanced chunking using enhanced DocumentProcessor."""
        # Use the enhanced DocumentProcessor with new chunking methods
        processor = DocumentProcessor(str(self.docs_directory))
        
        print(f"Using chunking method: {self.chunking_method}")
        print(f"Chunk size: {self.chunk_size}")
        
        # Process documents with the specified chunking method
        all_chunks = processor.process_documents_with_chunking(
            chunking_method=self.chunking_method,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap
        )
        
        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

def create_vector_database(config: Dict[str, Any], texts_dir: str):
    """Create new vector database with specified configuration."""
    
    print(f"Creating vector database with configuration:")
    print(f"  Chunk size: {config.get('chunk_size', 500)}")
    print(f"  Chunk overlap: {config.get('chunk_overlap', 50)}")
    print(f"  Chunking method: {config.get('chunking_method', 'paragraph')}")
    print(f"  Embedding model: {config.get('embedding_model', 'text-embedding-3-small')}")
    print(f"  Embed chapter info: {config.get('embed_chapter_info', True)}")
    
    # Initialize enhanced processor
    processor = AdvancedDocumentProcessor(texts_dir, config)
    
    # Process documents
    chunks = processor.process_documents()
    
    if not chunks:
        print("❌ No chunks created. Check your documents directory.")
        return
    
    # Initialize RAG system with custom embedding model
    rag_system = RAGSystem(
        embedding_model=config.get('embedding_model', 'text-embedding-3-small'),
        language_model=config.get('language_model', 'gpt-4o-mini'),
        vector_db_path=config.get('vector_db_path', 'models/vector_db.pkl')
    )
    
    # Build database
    print(f"\nBuilding vector database...")
    added = rag_system.build_database_from_chunks(chunks)
    print(f"Added {added}/{len(chunks)} chunks to vector database")
    
    # Save database
    if rag_system.save_database():
        print(f"✅ Vector database saved to: {rag_system.vector_db_path}")
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'config': config,
            'total_chunks': added,
            'source_files': list(set(chunk['metadata']['file_path'] for chunk in chunks)),
            'chunking_stats': {
                'method': config.get('chunking_method', 'paragraph'),
                'avg_chunk_size': sum(len(chunk['text']) for chunk in chunks) / len(chunks),
                'chunk_size_config': config.get('chunk_size', 500),
                'overlap_config': config.get('chunk_overlap', 50)
            }
        }
        
        metadata_path = Path(rag_system.vector_db_path).with_suffix('.metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"📊 Metadata saved to: {metadata_path}")
    else:
        print("❌ Failed to save vector database")

def main():
    parser = argparse.ArgumentParser(description='Create new vector database with configurable chunking')
    parser.add_argument('--config', required=True, help='Path to create_vdb_config.json')
    parser.add_argument('--text-inputs', required=True, help='Path to texts directory')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.text_inputs):
        print(f"❌ Texts directory not found: {args.text_inputs}")
        sys.exit(1)
    
    # Load configuration
    try:
        config = load_vdb_config(args.config)
        print(f"Loaded config: {args.config}")
        
        # Create vector database
        create_vector_database(config, args.text_inputs)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
