#!/usr/bin/env python3
"""Test script for enhanced RAG system with section-based chunking."""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'models'))

from document_processor import DocumentProcessor
from rag_system import RAGSystem
import json

def test_enhanced_rag_system():
    """Test the complete RAG pipeline with enhanced chunking."""
    print("Testing Enhanced RAG System with Section-Based Chunking")
    print("=" * 60)
    
    # Step 1: Create enhanced chunks
    print("1. Creating document chunks with section-based chunking...")
    processor = DocumentProcessor()
    chunks = processor.process_documents_with_chunking(
        chunking_method='section',
        chunk_size=800,
        overlap=100
    )
    print(f"   Created {len(chunks)} chunks using section-based chunking")
    
    # Show sample chunk metadata
    if chunks:
        sample_chunk = chunks[0]
        print(f"   Sample chunk metadata keys: {list(sample_chunk['metadata'].keys())}")
        if 'chapter_title' in sample_chunk['metadata']:
            print(f"   Sample chapter: {sample_chunk['metadata']['chapter_title']}")
        if 'section_title' in sample_chunk['metadata']:
            print(f"   Sample section: {sample_chunk['metadata']['section_title']}")
    
    # Step 2: Initialize RAG system
    print("\n2. Initializing RAG system...")
    rag = RAGSystem(vector_db_path="models/vector_db_section_enhanced.pkl")
    print(f"   OpenAI available: {rag.openai_available}")
    
    # Step 3: Build vector database (test with subset for speed)
    print("\n3. Building vector database...")
    test_chunks = chunks[:50]  # Use first 50 chunks for testing
    print(f"   Testing with {len(test_chunks)} chunks...")
    
    if rag.openai_available:
        added_count = rag.build_database_from_chunks(test_chunks)
        print(f"   Successfully added {added_count}/{len(test_chunks)} chunks to vector database")
        
        # Save the database
        if rag.save_database():
            print("   Vector database saved successfully")
        
        # Step 4: Test queries
        print("\n4. Testing RAG queries...")
        test_queries = [
            "太陽病的主要症狀是什麼？",
            "中風和傷寒有什麼區別？",
            "桂枝湯的組成是什麼？"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: {query}")
            try:
                result = rag.query(query, top_n=3)
                print(f"   Response length: {len(result['response'])} characters")
                print(f"   Retrieved {result['num_chunks_used']} relevant chunks")
                
                # Show retrieved chunk info
                for j, chunk_info in enumerate(result['retrieved_chunks']):
                    similarity = chunk_info['similarity']
                    metadata = chunk_info['metadata']
                    chunk_title = metadata.get('chunk_title', 'Unknown')
                    print(f"     Chunk {j+1}: {chunk_title} (similarity: {similarity:.3f})")
                
            except Exception as e:
                print(f"   Error with query: {e}")
    else:
        print("   OpenAI API not available - testing chunk structure only")
        
        # Test chunk structure without API calls
        print("   Analyzing chunk structure...")
        chapter_chunks = [c for c in test_chunks if c['metadata'].get('chunk_type') == 'section']
        section_chunks = [c for c in test_chunks if 'section_title' in c['metadata']]
        
        print(f"   Section-based chunks: {len(section_chunks)}")
        print(f"   Chunks with chapter info: {len([c for c in test_chunks if 'chapter_title' in c['metadata']])}")
        
        # Show sample chunk content
        if section_chunks:
            sample = section_chunks[0]
            print(f"   Sample section chunk:")
            print(f"     Title: {sample['metadata'].get('chunk_title', 'N/A')}")
            print(f"     Chapter: {sample['metadata'].get('chapter_title', 'N/A')}")
            print(f"     Section: {sample['metadata'].get('section_title', 'N/A')}")
            print(f"     Content length: {len(sample['text'])} characters")
    
    # Step 5: Database info
    print("\n5. Vector database information:")
    db_info = rag.get_database_info()
    for key, value in db_info.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Enhanced RAG system test completed!")
    
    return {
        'total_chunks': len(chunks),
        'test_chunks': len(test_chunks),
        'openai_available': rag.openai_available,
        'database_info': db_info
    }

if __name__ == "__main__":
    results = test_enhanced_rag_system()
    
    # Save test results
    with open('test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nTest results saved to test_results.json")
