#!/usr/bin/env python3
"""Test script for enhanced TCM document chunking."""

import sys
import os
sys.path.append(str(Path(__file__).parent.parent / "models"))

from document_processor import DocumentProcessor

def test_chunking_methods():
    """Test the new chunking methods."""
    print("Testing Enhanced TCM Document Chunking")
    print("=" * 50)
    
    # Initialize document processor
    processor = DocumentProcessor()
    
    # Test with a sample TCM text
    sample_text = """《人紀傷寒論》

倪海廈

#CHAPTER
《傷寒論》序言
經方在我國漢朝以前其實就已經存在，應該是來自西域，而且已經在我國流傳千年，只是經方在漢朝之前是沒有正式的辨症依據，用來指導如何使用經方，所以知道如何正確使用經方的醫師非常之少。

#CHAPTER
前言    
在《難經》中有五十八難曰：『傷寒有幾？其脈有變否？然。傷寒有五，有中風，有傷寒，有濕溫，有熱病，有溫病，其所苦各不同。』傷寒這兩字，就廣義的定義來說，應該說是『傷於寒』。

#CHAPTER
辨太陽病脈證並治法上篇

#SECTION
一．「太陽」之為病，脈浮，頭項強痛而惡寒。    
對於「太陽」，可以代表經絡中的太陽經，也可以代表全體陰陽狀態的太陽。八綱辨症中表裏的表，在這裡指的就是太陽。

#SECTION     
二．「太陽病」 發熱，汗出，惡風，脈緩者，名為「中風」。    
太陽病的中風證，就有下面的症狀，「發熱，汗出，惡風，脈緩」，如果腸胃功能很好，體力很好，也會有得到感冒的時候。

#FORMULA
桂枝湯
桂枝三兩，芍藥三兩，炙甘草二兩，生薑三兩，大棗十二枚。
上五味，㕮咀三味，以水七升，微火煮取三升，去滓，適寒溫，服一升。
"""
    
    print("1. Testing marker detection:")
    markers = processor.detect_tcm_markers(sample_text)
    print(f"   Chapters found: {len(markers['chapters'])}")
    for i, chapter in enumerate(markers['chapters']):
        print(f"     Chapter {i+1}: {chapter['title']}")
    
    print(f"   Sections found: {len(markers['sections'])}")
    for i, section in enumerate(markers['sections']):
        print(f"     Section {i+1}: {section['title']}")
    
    print(f"   Formulas found: {len(markers['formulas'])}")
    for i, formula in enumerate(markers['formulas']):
        print(f"     Formula {i+1}: {formula['title']}")
    
    print("\n2. Testing chapter-based chunking:")
    chapter_chunks = processor.chunk_by_chapters(sample_text, max_chunk_size=1000)
    print(f"   Created {len(chapter_chunks)} chapter-based chunks")
    for i, chunk in enumerate(chapter_chunks):
        print(f"     Chunk {i+1}: {chunk['title']} ({len(chunk['text'])} chars)")
    
    print("\n3. Testing section-based chunking:")
    section_chunks = processor.chunk_by_sections(sample_text, max_chunk_size=800)
    print(f"   Created {len(section_chunks)} section-based chunks")
    for i, chunk in enumerate(section_chunks):
        print(f"     Chunk {i+1}: {chunk['title']} ({len(chunk['text'])} chars)")
    
    print("\n4. Testing document processing with different methods:")
    methods = ['regular', 'chapter', 'section']
    for method in methods:
        print(f"\n   Testing {method} chunking method:")
        try:
            chunks = processor.process_documents_with_chunking(
                chunking_method=method,
                chunk_size=800,
                overlap=100
            )
            print(f"     Successfully created {len(chunks)} chunks using {method} method")
            if chunks:
                print(f"     Sample chunk metadata: {list(chunks[0]['metadata'].keys())}")
        except Exception as e:
            print(f"     Error with {method} method: {e}")
    
    print("\n" + "=" * 50)
    print("Enhanced chunking test completed!")

if __name__ == "__main__":
    test_chunking_methods()
