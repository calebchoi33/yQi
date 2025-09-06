#!/usr/bin/env python3
"""Test script for semantic chunking strategies."""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "models"))

from chunking_strategies_semantic import SemanticSectionChunkingStrategy, SemanticChapterChunkingStrategy, HybridChunkingStrategy
from document_processor import DocumentProcessor
import json

def test_semantic_chunking():
    """Test semantic chunking strategies with actual TCM documents."""
    print("Testing Semantic Chunking Strategies")
    print("=" * 50)
    
    # Initialize document processor
    processor = DocumentProcessor()
    
    # Test different semantic strategies
    strategies = {
        'semantic_section': SemanticSectionChunkingStrategy(),
        'semantic_chapter': SemanticChapterChunkingStrategy(),
        'hybrid': HybridChunkingStrategy()
    }
    
    results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"\n{strategy_name.upper()} CHUNKING:")
        print("-" * 30)
        
        chunks = processor.process_documents_with_chunking(
            chunking_method=strategy_name,
            chunk_size=1500  # This will be ignored for semantic strategies
        )
        
        if not chunks:
            print("  No chunks created")
            continue
            
        # Analyze chunk characteristics
        chunk_sizes = [len(chunk['text']) for chunk in chunks]
        semantic_chunks = [c for c in chunks if c.get('preserved_complete', False)]
        
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Semantic chunks (complete): {len(semantic_chunks)}")
        print(f"  Chunk sizes:")
        print(f"    Min: {min(chunk_sizes):,} chars")
        print(f"    Max: {max(chunk_sizes):,} chars") 
        print(f"    Avg: {sum(chunk_sizes)//len(chunk_sizes):,} chars")
        print(f"    Chunks > 1000: {sum(1 for s in chunk_sizes if s > 1000)}")
        print(f"    Chunks > 2000: {sum(1 for s in chunk_sizes if s > 2000)}")
        print(f"    Chunks > 5000: {sum(1 for s in chunk_sizes if s > 5000)}")
        
        # Show sample chunk metadata
        if chunks:
            sample = chunks[0]
            print(f"  Sample chunk type: {sample.get('type', 'unknown')}")
            print(f"  Sample title: {sample.get('title', 'No title')[:50]}...")
            if 'chapter_title' in sample:
                print(f"  Chapter: {sample['chapter_title']}")
            if 'section_title' in sample:
                print(f"  Section: {sample['section_title']}")
        
        results[strategy_name] = {
            'total_chunks': len(chunks),
            'semantic_chunks': len(semantic_chunks),
            'min_size': min(chunk_sizes),
            'max_size': max(chunk_sizes),
            'avg_size': sum(chunk_sizes) // len(chunk_sizes),
            'large_chunks': sum(1 for s in chunk_sizes if s > 2000)
        }
    
    # Compare with regular chunking
    print(f"\nREGULAR CHUNKING (for comparison):")
    print("-" * 30)
    
    regular_chunks = processor.process_documents_with_chunking(
        chunking_method='regular',
        chunk_size=1000
    )
    
    if regular_chunks:
        regular_sizes = [len(chunk['text']) for chunk in regular_chunks]
        print(f"  Total chunks: {len(regular_chunks)}")
        print(f"  Avg size: {sum(regular_sizes)//len(regular_sizes):,} chars")
        print(f"  All chunks <= 1000: {all(s <= 1200 for s in regular_sizes)}")  # Allow some overlap
        
        results['regular'] = {
            'total_chunks': len(regular_chunks),
            'avg_size': sum(regular_sizes) // len(regular_sizes)
        }
    
    # Save detailed results
    output_path = Path(__file__).parent.parent / "output" / "semantic_chunking_test.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n" + "=" * 50)
    print("Semantic chunking test completed!")
    print(f"Results saved to: {output_path}")
    
    return results

if __name__ == "__main__":
    test_semantic_chunking()
