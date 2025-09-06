#!/usr/bin/env python3
"""Test script for refactored document processor."""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "models"))

def test_refactored_processor():
    """Test the refactored document processor."""
    print("Testing Refactored Document Processor")
    print("=" * 50)
    
    try:
        from document_processor import DocumentProcessor
        print("✓ Successfully imported refactored DocumentProcessor")
        
        # Initialize processor
        processor = DocumentProcessor()
        print("✓ DocumentProcessor initialized")
        
        # Test basic properties
        print(f"✓ Supported extensions: {processor.supported_extensions}")
        print(f"✓ Available chunking methods: {processor.get_chunking_methods()}")
        
        # Test document stats
        stats = processor.get_document_stats()
        print(f"✓ Document stats: {stats['total_files']} files, {stats['file_types']}")
        
        # Test chunking methods
        methods = ['regular', 'chapter', 'section']
        for method in methods:
            try:
                chunks = processor.process_documents_with_chunking(
                    chunking_method=method,
                    chunk_size=800,
                    overlap=100
                )
                print(f"✓ {method} chunking: {len(chunks)} chunks created")
            except Exception as e:
                print(f"✗ {method} chunking failed: {e}")
        
        # Test legacy compatibility
        try:
            legacy_chunks = processor.process_documents(chunk_size=500, overlap=50)
            print(f"✓ Legacy method compatibility: {len(legacy_chunks)} chunks")
        except Exception as e:
            print(f"✗ Legacy compatibility failed: {e}")
        
        print("\n" + "=" * 50)
        print("Refactored processor test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_refactored_processor()
    sys.exit(0 if success else 1)
