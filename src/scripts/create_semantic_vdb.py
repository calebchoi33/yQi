#!/usr/bin/env python3
"""Create semantic vector database with complete section preservation."""

import json
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add models directory to path
sys.path.append(str(Path(__file__).parent.parent / 'models'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(str(Path(__file__).parent.parent.parent / 'evaluation' / '.env'))

from rag_system import RAGSystem
from document_processor import DocumentProcessor

def create_semantic_vector_database(config_path: str, output_path: Optional[str] = None):
    """Create a semantic vector database that preserves complete TCM sections."""
    
    print("🧠 Creating Semantic Vector Database")
    print("=" * 50)
    
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    chunking_method = config.get('chunking_method', 'semantic_section')
    embedding_model = config.get('embedding_model', 'text-embedding-3-large')
    chunk_size = config.get('chunk_size', 1500)  # Used as fallback only
    
    print(f"📋 Configuration:")
    print(f"   Chunking method: {chunking_method}")
    print(f"   Embedding model: {embedding_model}")
    print(f"   Fallback chunk size: {chunk_size}")
    print(f"   Preserve semantic boundaries: {config.get('preserve_semantic_boundaries', True)}")
    
    # Initialize document processor
    processor = DocumentProcessor()
    
    # Create semantic chunks
    print(f"\n📚 Processing TCM documents...")
    chunks = processor.process_documents_with_chunking(
        chunking_method=chunking_method,
        chunk_size=chunk_size
    )
    
    if not chunks:
        print("❌ No chunks created. Check your documents directory.")
        return False
    
    # Analyze semantic preservation
    semantic_chunks = [c for c in chunks if c.get('preserved_complete', False)]
    chunk_sizes = [len(chunk['text']) for chunk in chunks]
    
    print(f"\n📊 Chunk Analysis:")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Semantic chunks (complete): {len(semantic_chunks)}")
    print(f"   Semantic preservation: {len(semantic_chunks)/len(chunks)*100:.1f}%")
    print(f"   Average chunk size: {sum(chunk_sizes)//len(chunk_sizes):,} chars")
    print(f"   Largest chunk: {max(chunk_sizes):,} chars")
    print(f"   Chunks > 2000 chars: {sum(1 for s in chunk_sizes if s > 2000)}")
    
    # Initialize RAG system with large embeddings
    print(f"\n🔗 Initializing RAG system...")
    rag_system = RAGSystem(
        embedding_model=embedding_model,
        vector_db_path=output_path
    )
    
    if not rag_system.openai_available:
        print("⚠️  OpenAI API not available. Vector database created without embeddings.")
        print("   Set OPENAI_API_KEY environment variable to enable embeddings.")
    
    # Build vector database
    print(f"\n🏗️  Building vector database...")
    success_count = rag_system.build_database_from_chunks(chunks)
    success = success_count > 0
    
    if success:
        # Save metadata about the semantic database
        metadata = {
            'created_at': datetime.now().isoformat(),
            'chunking_method': chunking_method,
            'embedding_model': embedding_model,
            'total_chunks': len(chunks),
            'semantic_chunks': len(semantic_chunks),
            'semantic_preservation_ratio': len(semantic_chunks) / len(chunks),
            'avg_chunk_size': sum(chunk_sizes) // len(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'large_chunks_count': sum(1 for s in chunk_sizes if s > 2000),
            'config_used': config
        }
        
        metadata_path = Path(output_path).with_suffix('.metadata.json') if output_path else Path('models/semantic_vector_db.metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Semantic vector database created successfully!")
        print(f"   Database: {rag_system.vector_db_path}")
        print(f"   Metadata: {metadata_path}")
        print(f"   Embeddings: {'✓' if rag_system.openai_available else '✗'}")
        
        # Show sample semantic chunks
        print(f"\n📝 Sample semantic chunks:")
        for i, chunk in enumerate(semantic_chunks[:3]):
            print(f"   {i+1}. {chunk.get('title', 'No title')[:60]}...")
            print(f"      Type: {chunk.get('type', 'unknown')}")
            print(f"      Size: {len(chunk['text']):,} chars")
            if 'chapter_title' in chunk:
                print(f"      Chapter: {chunk['chapter_title']}")
            if 'section_title' in chunk:
                print(f"      Section: {chunk['section_title']}")
        
        return True
    else:
        print("❌ Failed to create vector database")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create semantic vector database')
    parser.add_argument('--config', 
                       default='config/create_vdb_config_semantic.json',
                       help='Configuration file path')
    parser.add_argument('--output',
                       default='models/semantic_vector_db.pkl',
                       help='Output vector database path')
    
    args = parser.parse_args()
    
    try:
        success = create_semantic_vector_database(args.config, args.output)
        if success:
            print(f"\n🎉 Ready to use semantic RAG system!")
            print(f"   Use 'semantic_section' chunking for best TCM concept preservation")
            print(f"   Recommended: text-embedding-3-large for superior accuracy")
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
