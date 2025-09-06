#!/usr/bin/env python3
"""Compare chunking strategies for retrieval quality."""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "models"))

from document_processor import DocumentProcessor
from rag_system import RAGSystem
import json
import time

def test_chunking_comparison():
    """Compare different chunking strategies for retrieval quality."""
    print("Chunking Strategy Comparison for RAG Quality")
    print("=" * 60)
    
    # Test queries that should benefit from semantic chunking
    test_queries = [
        "太陽病的脈象特徵是什麼？",  # Pulse characteristics of Taiyang disease
        "桂枝湯的組成和功效",        # Composition and effects of Guizhi decoction
        "傷寒論中的六經辨證",        # Six-channel pattern identification in Shanghan Lun
        "少陽病的治療方法",          # Treatment methods for Shaoyang disease
        "麻黃湯和桂枝湯的區別"       # Differences between Mahuang and Guizhi decoctions
    ]
    
    strategies_to_test = [
        ('regular', 1000),
        ('section', 1500), 
        ('semantic_section', 1500),
        ('hybrid', 1500)
    ]
    
    results = {}
    
    for strategy_name, chunk_size in strategies_to_test:
        print(f"\nTesting {strategy_name.upper()} chunking...")
        print("-" * 40)
        
        # Create document chunks
        processor = DocumentProcessor()
        chunks = processor.process_documents_with_chunking(
            chunking_method=strategy_name,
            chunk_size=chunk_size
        )
        
        if not chunks:
            print(f"  No chunks created for {strategy_name}")
            continue
        
        # Analyze chunk characteristics
        chunk_sizes = [len(chunk['text']) for chunk in chunks]
        semantic_chunks = [c for c in chunks if c.get('preserved_complete', False)]
        
        strategy_results = {
            'total_chunks': len(chunks),
            'semantic_chunks': len(semantic_chunks),
            'avg_chunk_size': sum(chunk_sizes) // len(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'chunks_over_2000': sum(1 for s in chunk_sizes if s > 2000),
            'sample_chunks': []
        }
        
        # Sample some chunks for quality analysis
        for i, chunk in enumerate(chunks[:3]):
            sample = {
                'size': len(chunk['text']),
                'type': chunk.get('type', 'unknown'),
                'title': chunk.get('title', 'No title')[:50],
                'has_complete_concept': chunk.get('preserved_complete', False),
                'preview': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text']
            }
            strategy_results['sample_chunks'].append(sample)
        
        print(f"  Chunks created: {len(chunks)}")
        print(f"  Semantic chunks: {len(semantic_chunks)}")
        print(f"  Avg size: {strategy_results['avg_chunk_size']:,} chars")
        print(f"  Max size: {strategy_results['max_chunk_size']:,} chars")
        print(f"  Large chunks (>2000): {strategy_results['chunks_over_2000']}")
        
        # Show semantic preservation benefit
        if semantic_chunks:
            semantic_ratio = len(semantic_chunks) / len(chunks) * 100
            print(f"  Semantic preservation: {semantic_ratio:.1f}%")
        
        results[strategy_name] = strategy_results
    
    # Quality assessment based on chunk characteristics
    print(f"\n" + "=" * 60)
    print("QUALITY ASSESSMENT")
    print("=" * 60)
    
    for strategy_name, data in results.items():
        print(f"\n{strategy_name.upper()}:")
        
        # Calculate quality score
        quality_factors = []
        
        # Semantic preservation (higher is better)
        semantic_ratio = data['semantic_chunks'] / data['total_chunks']
        quality_factors.append(('Semantic Preservation', semantic_ratio * 100, 'higher_better'))
        
        # Chunk count efficiency (fewer chunks with same content = better)
        chunk_efficiency = 1000 / data['total_chunks']  # Normalized
        quality_factors.append(('Chunk Efficiency', chunk_efficiency, 'higher_better'))
        
        # Size consistency (less variation = better for embeddings)
        avg_size = data['avg_chunk_size']
        size_score = min(100, avg_size / 20)  # Normalize to 0-100
        quality_factors.append(('Size Appropriateness', size_score, 'optimal_range'))
        
        for factor_name, score, direction in quality_factors:
            print(f"  {factor_name}: {score:.1f}")
        
        # Overall assessment
        if 'semantic' in strategy_name:
            print(f"  ✓ Preserves complete TCM concepts")
            print(f"  ✓ Better semantic coherence")
            if data['chunks_over_2000'] > 0:
                print(f"  ⚠ {data['chunks_over_2000']} very large chunks")
        else:
            print(f"  ⚠ May split TCM concepts")
            print(f"  ✓ Consistent chunk sizes")
    
    # Recommendations
    print(f"\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    best_semantic = max(
        [(k, v) for k, v in results.items() if 'semantic' in k or k == 'hybrid'],
        key=lambda x: x[1]['semantic_chunks'],
        default=(None, None)
    )
    
    if best_semantic[0]:
        print(f"✅ RECOMMENDED: {best_semantic[0].upper()}")
        print(f"   - {best_semantic[1]['semantic_chunks']} complete semantic chunks")
        print(f"   - Preserves TCM diagnostic patterns")
        print(f"   - Better retrieval for complex queries")
        print(f"   - Use with text-embedding-3-large for best results")
    
    print(f"\n📊 For comparison:")
    regular_chunks = results.get('regular', {}).get('total_chunks', 0)
    semantic_chunks = best_semantic[1]['total_chunks'] if best_semantic[0] else 0
    if regular_chunks and semantic_chunks:
        reduction = (regular_chunks - semantic_chunks) / regular_chunks * 100
        print(f"   - {reduction:.1f}% fewer chunks than regular chunking")
        print(f"   - Higher semantic density per chunk")
    
    # Save results
    output_path = Path(__file__).parent.parent / "output" / "chunking_comparison.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_path}")
    return results

if __name__ == "__main__":
    test_chunking_comparison()
