#!/usr/bin/env python3
"""Debug script to check document processing paths."""

import sys
from pathlib import Path

# Add models to path
sys.path.append(str(Path(__file__).parent))

from document_processor import DocumentProcessor

def main():
    print("🔍 Debugging Document Processor")
    print("=" * 40)
    
    processor = DocumentProcessor()
    
    print(f"📁 Docs directory: {processor.docs_directory}")
    print(f"📁 Absolute path: {processor.docs_directory.absolute()}")
    print(f"✅ Directory exists: {processor.docs_directory.exists()}")
    
    if processor.docs_directory.exists():
        print(f"\n📄 Files in directory:")
        for file_path in processor.docs_directory.iterdir():
            if file_path.is_file():
                print(f"   - {file_path.name} ({file_path.suffix})")
        
        print(f"\n🎯 Supported extensions: {processor.supported_extensions}")
        
        available_docs = processor.get_available_documents()
        print(f"\n📋 Available documents: {available_docs}")
        
        if available_docs:
            print(f"\n🧪 Testing document processing...")
            chunks = processor.process_documents(chunk_size=200, overlap=20)
            print(f"📝 Generated {len(chunks)} chunks")
            
            if chunks:
                sample = chunks[0]
                print(f"\n📋 Sample chunk:")
                print(f"   Source: {sample['source']}")
                print(f"   Text: {sample['text'][:100]}...")
        else:
            print("❌ No supported documents found")
    else:
        print("❌ Directory does not exist")

if __name__ == "__main__":
    main()
