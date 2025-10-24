#!/usr/bin/env python3
"""
Tag Statistics Analyzer

Takes tagging/output/book_name_tags.json and tags.json and outputs statistics 
on the number of tags per section:
- mean
- min
- max
- 50th, 75th, 90th percentiles
"""

import json
import argparse
import numpy as np
from pathlib import Path
import sys


def load_json_file(file_path):
    """Load and parse JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{file_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")
        sys.exit(1)


def extract_tag_counts_by_category(data):
    """Extract tag counts by category from the data structure."""
    category_counts = {}
    
    if 'sections' not in data:
        print("Warning: Unexpected data format. Looking for 'sections' key.")
        return category_counts
    
    # Find all possible tag categories
    all_categories = set()
    for section in data['sections']:
        for key in section.keys():
            if key.endswith('_tags'):
                all_categories.add(key)
    
    # Initialize category counts
    for category in all_categories:
        category_counts[category] = []
    
    # Extract counts for each category
    for section in data['sections']:
        for category in all_categories:
            if category in section:
                tag_count = len(section[category])
                category_counts[category].append(tag_count)
            else:
                # Section doesn't have this category, count as 0
                category_counts[category].append(0)
    
    return category_counts


def calculate_statistics(tag_counts):
    """Calculate statistics for tag counts."""
    if not tag_counts:
        return None
    
    tag_counts = np.array(tag_counts)
    
    stats = {
        'count': len(tag_counts),
        'total_tags': np.sum(tag_counts),
        'mean': np.mean(tag_counts),
        'min': np.min(tag_counts),
        'max': np.max(tag_counts),
        'median': np.median(tag_counts),  # 50th percentile
        'p75': np.percentile(tag_counts, 75),
        'p90': np.percentile(tag_counts, 90),
        'std': np.std(tag_counts)
    }
    
    return stats


def print_statistics(category_stats, file_name):
    """Print formatted statistics for all categories."""
    if not category_stats:
        print(f"No data found in {file_name}")
        return
    
    print(f"\n=== Tag Statistics for {file_name} ===")
    
    # Print overall summary
    total_sections = None
    for category, stats in category_stats.items():
        if total_sections is None:
            total_sections = stats['count']
        print(f"\n--- {category.replace('_tags', '').title()} Tags ---")
        print(f"Total sections: {stats['count']}")
        print(f"Total tags: {stats['total_tags']}")
        print(f"Mean tags per section: {stats['mean']:.2f}")
        print(f"Min tags per section: {stats['min']}")
        print(f"Max tags per section: {stats['max']}")
        print(f"Median (50th percentile): {stats['median']:.2f}")
        print(f"75th percentile: {stats['p75']:.2f}")
        print(f"90th percentile: {stats['p90']:.2f}")
        print(f"Standard deviation: {stats['std']:.2f}")
    
    # Print summary table
    print(f"\n--- Summary Table ---")
    print(f"{'Category':<20} {'Mean':<8} {'Min':<5} {'Max':<5} {'Median':<8} {'P75':<6} {'P90':<6}")
    print("-" * 70)
    for category, stats in category_stats.items():
        category_name = category.replace('_tags', '').title()
        print(f"{category_name:<20} {stats['mean']:<8.2f} {stats['min']:<5} {stats['max']:<5} {stats['median']:<8.2f} {stats['p75']:<6.2f} {stats['p90']:<6.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze tag statistics from JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tag_statistics.py tagging/output/book_tags.json
  python tag_statistics.py diagnosis_inference/workflows/chunk_rag/data/tags.json
  python tag_statistics.py file1.json file2.json
        """
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        help='JSON files to analyze (book_tags.json, tags.json, etc.)'
    )
    
    args = parser.parse_args()
    
    all_stats = {}
    
    for file_path in args.files:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"Warning: File '{file_path}' does not exist, skipping.")
            continue
        
        print(f"\nProcessing: {file_path}")
        data = load_json_file(file_path)
        category_counts = extract_tag_counts_by_category(data)
        
        if category_counts:
            category_stats = {}
            for category, counts in category_counts.items():
                if counts:  # Only process categories that have data
                    stats = calculate_statistics(counts)
                    if stats:
                        category_stats[category] = stats
            
            if category_stats:
                print_statistics(category_stats, file_path.name)
                all_stats[str(file_path)] = category_stats
            else:
                print(f"No valid tag data found in {file_path.name}")
        else:
            print(f"No tag data found in {file_path.name}")


if __name__ == "__main__":
    main()
