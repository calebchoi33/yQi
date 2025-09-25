#!/usr/bin/env python3
"""Structured Vector Database for TCM Knowledge Base with multi-vector columns."""

import json
import pickle
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import re
from datetime import datetime
# Removed tag_processor import - functionality integrated directly
import numpy as np
from dataclasses import dataclass, asdict
import re

logger = logging.getLogger(__name__)

@dataclass
class VectorRecord:
    """Represents a single row in the structured vector database."""
    # Content vectors
    symptoms_vector: Optional[List[float]] = None
    organs_vector: Optional[List[float]] = None
    formulas_vector: Optional[List[float]] = None
    patterns_vector: Optional[List[float]] = None
    full_content_vector: Optional[List[float]] = None
    
    # Metadata
    book_name: str = ""
    book_id: str = ""
    chapter_index: int = 0
    chapter_title: str = ""
    section_index: int = 0
    section_title: str = ""
    page_index: int = 0
    subsection_index: int = 0
    content_type: str = ""
    word_count: int = 0
    char_count: int = 0
    created_at: str = ""
    source_file: str = ""
    
    # Full content
    full_content: str = ""
    symptoms_content: str = ""
    organs_content: str = ""
    formulas_content: str = ""
    patterns_content: str = ""
    
    # Tag-related fields
    related_tags: List[Dict[str, str]] = field(default_factory=list)
    tag_count: int = 0

class StructuredVectorDB:
    """Structured Vector Database with chapter/section-based rows and multi-vector columns."""
    
    def __init__(self, config_path: str, openai_client=None):
        """Initialize structured vector database with configuration."""
        self.config_path = config_path
        self.openai_client = openai_client
        self.records: List[VectorRecord] = []
        self.metadata: Dict[str, Any] = {}
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Initialize vector columns configuration
        self.vector_columns = self.config.get('database_schema', {}).get('vector_columns', {})
        
        # Load tags data for tag embedding
        self.tags_data = self._load_tags_data()
        self.retrieval_settings = self.config['retrieval_settings']
        
        logger.info(f"Initialized StructuredVectorDB with {len(self.vector_columns)} vector columns")
        if self.tags_data:
            total_tags = sum(len(section.get('tags', [])) for section in self.tags_data.get('sections', []))
            logger.info(f"Loaded {total_tags} TCM tags for enhanced embedding")
    
    def _get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        """Generate embedding for text using OpenAI API."""
        if not self.openai_client:
            logger.warning("No OpenAI client available for embedding generation")
            return []
        if not text.strip():
            logger.warning("Empty text provided for embedding")
            return []
        
        # Truncate text if too long (approximate token limit)
        max_chars = 6000  # Conservative estimate for 8192 token limit
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            logger.warning(f"Text truncated to {max_chars} characters for embedding")
        
        try:
            response = self.openai_client.embeddings.create(
                model=model,
                input=text.strip()
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def _extract_specialized_content(self, full_text: str, extraction_type: str) -> str:
        """Extract specialized content based on type using LLM."""
        if not self.openai_client or extraction_type not in self.vector_columns:
            return ""
        
        prompt = self.vector_columns[extraction_type]['extraction_prompt']
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a TCM expert. Extract only the requested content type from the given text. Return only the extracted content, no explanations."},
                    {"role": "user", "content": f"{prompt}\n\nText:\n{full_text}"}
                ],
                max_tokens=500,
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error extracting {extraction_type} content: {e}")
            return ""
    
    def _parse_document_structure(self, content: str, book_config: Dict) -> List[Dict]:
        """Parse document into structured sections based on book configuration."""
        sections = []
        
        # Get structure markers
        markers = book_config.get('structure_markers', {})
        chapter_pattern = markers.get('chapter_pattern', '')
        section_pattern = markers.get('section_pattern', '')
        
        # Split content into chapters
        if chapter_pattern:
            chapter_splits = re.split(f'({chapter_pattern})', content)
        else:
            chapter_splits = [content]  # Single chapter if no pattern
        
        current_chapter_index = 0
        current_chapter_title = "Unknown Chapter"
        
        for i, chunk in enumerate(chapter_splits):
            if not chunk.strip():
                continue
                
            # Check if this chunk is a chapter header
            if chapter_pattern and re.match(chapter_pattern, chunk.strip()):
                current_chapter_index += 1
                current_chapter_title = chunk.strip()
                continue
            
            # Split chapter content into sections
            if section_pattern:
                section_splits = re.split(f'({section_pattern})', chunk)
            else:
                section_splits = [chunk]  # Single section if no pattern
            
            current_section_index = 0
            current_section_title = ""
            
            for j, section_chunk in enumerate(section_splits):
                if not section_chunk.strip():
                    continue
                
                # Check if this chunk is a section header
                if section_pattern and re.match(section_pattern, section_chunk.strip()):
                    current_section_index += 1
                    current_section_title = section_chunk.strip()
                    continue
                
                # This is actual content - split large sections
                if len(section_chunk.strip()) > 50:  # Only process substantial content
                    # Split very large sections into smaller chunks
                    max_section_size = 4000  # Conservative size for embedding
                    content_text = section_chunk.strip()
                    
                    if len(content_text) <= max_section_size:
                        sections.append({
                            'content': content_text,
                            'chapter_index': current_chapter_index,
                            'chapter_title': current_chapter_title,
                            'section_index': current_section_index,
                            'section_title': current_section_title,
                            'book_id': book_config['book_id'],
                            'book_name': book_config['book_name']
                        })
                    else:
                        # Split large section into smaller parts
                        subsection_index = 0
                        for k in range(0, len(content_text), max_section_size):
                            subsection_content = content_text[k:k + max_section_size]
                            if subsection_content.strip():
                                subsection_index += 1
                                sections.append({
                                    'content': subsection_content.strip(),
                                    'chapter_index': current_chapter_index,
                                    'chapter_title': current_chapter_title,
                                    'section_index': current_section_index,
                                    'section_title': f"{current_section_title} (Part {subsection_index})",
                                    'subsection_index': subsection_index,
                                    'book_id': book_config['book_id'],
                                    'book_name': book_config['book_name']
                                })
        
        return sections
    
    def build_from_documents(self) -> int:
        """Build the structured vector database from configured documents."""
        total_records = 0
        
        for book_config in self.config['content_structure']['books']:
            logger.info(f"Processing book: {book_config['book_name']}")
            
            # Get documents directory
            docs_dir = Path(book_config.get('documents_directory', '../data/documents'))
            if not docs_dir.exists():
                logger.warning(f"Documents directory not found: {docs_dir}")
                continue
            
            # Find all supported document files in the directory
            supported_extensions = ['.txt', '.docx', '.doc']
            source_files = []
            for ext in supported_extensions:
                source_files.extend(docs_dir.glob(f'*{ext}'))
            
            if not source_files:
                logger.warning(f"No supported documents found in {docs_dir}")
                continue
            
            logger.info(f"Found {len(source_files)} document(s): {[f.name for f in source_files]}")
            
            # Process each source file for this book
            for source_file in source_files:
                logger.info(f"Processing file: {source_file.name}")
                
                # Read document content based on file type
                try:
                    if source_file.suffix.lower() == '.txt':
                        with open(source_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                    elif source_file.suffix.lower() in ['.docx', '.doc']:
                        # Extract text from Word files using python-docx
                        try:
                            from docx import Document
                            doc = Document(str(source_file))
                            content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                        except ImportError:
                            logger.error(f"python-docx not available for processing {source_file.name}")
                            continue
                        except Exception as docx_error:
                            logger.error(f"Error extracting text from {source_file.name}: {docx_error}")
                            continue
                    else:
                        logger.warning(f"Unsupported file type: {source_file.suffix}")
                        continue
                except Exception as e:
                    logger.error(f"Error reading {source_file}: {e}")
                    continue
                
                # Parse document structure
                sections = self._parse_document_structure(content, book_config)
                logger.info(f"Found {len(sections)} sections in {source_file.name}")
                
                # Process each section
                for section_data in sections:
                    record = self._create_vector_record(section_data, str(source_file))
                    if record:
                        self.records.append(record)
                        total_records += 1
                        
                        # Report tag embedding progress
                        if record.related_tags:
                            logger.info(f"Record {total_records}: Embedded {len(record.related_tags)} tags - {[tag['tag_zh'] for tag in record.related_tags[:3]]}")
                        
                        if total_records % 10 == 0:
                            logger.info(f"Processed {total_records} records...")
        
        # Generate final report
        tag_stats = self._generate_tag_embedding_report()
        logger.info(f"Built structured vector database with {total_records} records")
        logger.info(f"Tag Embedding Summary: {tag_stats}")
        return total_records
    
    def _generate_tag_embedding_report(self) -> Dict[str, Any]:
        """Generate a report on tag embedding statistics."""
        total_records = len(self.records)
        records_with_tags = sum(1 for record in self.records if record.related_tags)
        total_tags_embedded = sum(len(record.related_tags) for record in self.records)
        
        # Count unique tags used
        unique_tags = set()
        for record in self.records:
            for tag in record.related_tags:
                unique_tags.add(tag['tag_zh'])
        
        # Find most common tags
        tag_counts = {}
        for record in self.records:
            for tag in record.related_tags:
                tag_zh = tag['tag_zh']
                tag_counts[tag_zh] = tag_counts.get(tag_zh, 0) + 1
        
        most_common = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        report = {
            'total_records': total_records,
            'records_with_tags': records_with_tags,
            'tag_coverage_percentage': round((records_with_tags / total_records * 100), 2) if total_records > 0 else 0,
            'total_tags_embedded': total_tags_embedded,
            'unique_tags_used': len(unique_tags),
            'avg_tags_per_record': round(total_tags_embedded / total_records, 2) if total_records > 0 else 0,
            'most_common_tags': most_common[:5]
        }
        
        return report
    
    def _create_vector_record(self, section_data: Dict, source_file: str) -> Optional[VectorRecord]:
        """Create a vector record from section data with embedded tags."""
        full_content = section_data['content']
        
        # Find and embed related tags
        related_tags = self._find_related_tags(section_data)
        tag_enhanced_content = self._enhance_content_with_tags(full_content, related_tags)
        
        # Create base record
        record = VectorRecord(
            book_name=section_data['book_name'],
            book_id=section_data['book_id'],
            chapter_index=section_data['chapter_index'],
            chapter_title=section_data['chapter_title'],
            section_index=section_data['section_index'],
            section_title=section_data['section_title'],
            content_type="section",
            word_count=len(full_content.split()),
            char_count=len(full_content),
            created_at=datetime.now().isoformat(),
            source_file=source_file,
            full_content=tag_enhanced_content
        )
        
        # Store related tags in record
        record.related_tags = related_tags
        record.tag_count = len(related_tags)
        
        # Generate specialized content extractions with tag enhancement
        for vector_type in self.vector_columns.keys():
            if vector_type == 'full_content_vector':
                extracted_content = tag_enhanced_content
            else:
                extracted_content = self._extract_specialized_content(tag_enhanced_content, vector_type)
            
            # Store extracted content
            setattr(record, vector_type.replace('_vector', '_content'), extracted_content)
            
            # Generate embedding
            embedding = self._get_embedding(
                extracted_content, 
                self.vector_columns[vector_type]['embedding_model']
            )
            setattr(record, vector_type, embedding)
        
        return record
    
    def _load_tags_data(self) -> Dict[str, Any]:
        """Load tags data from tags.json file."""
        tags_file = "../data/tags.json"
        try:
            with open(tags_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading tags data: {e}")
            return {}
    
    def _find_related_tags(self, section_data: Dict) -> List[Dict[str, str]]:
        """Find tags related to the current section using LLM analysis."""
        if not self.tags_data or not self.openai_client:
            return []
        
        try:
            # Get all available tags
            all_tags = []
            for section in self.tags_data.get('sections', []):
                for tag in section.get('tags', []):
                    if tag.get('tag_zh') and tag.get('tag_en'):
                        all_tags.append(tag)
            
            if not all_tags:
                return []
            
            # Create tag reference string
            tag_list = "\n".join([f"- {tag['tag_zh']} ({tag['tag_en']})" for tag in all_tags[:100]])  # Limit for token constraints
            
            content = section_data['content'][:2000]  # Limit content length
            
            # Query LLM to find related tags
            prompt = f"""Analyze this Traditional Chinese Medicine text and identify which tags are most relevant:

TEXT:
{content}

AVAILABLE TAGS:
{tag_list}

Return ONLY the Chinese tag names (tag_zh) that are directly relevant to this text, separated by commas. Maximum 10 tags.
Example format: 頭痛,眩暈,失眠,脈弦細"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            selected_tag_names = response.choices[0].message.content.strip().split(',')
            selected_tag_names = [name.strip() for name in selected_tag_names if name.strip()]
            
            # Find matching tag objects
            related_tags = []
            for tag in all_tags:
                if tag['tag_zh'] in selected_tag_names:
                    related_tags.append(tag)
            
            logger.debug(f"Found {len(related_tags)} related tags for section")
            return related_tags
            
        except Exception as e:
            logger.warning(f"Error finding related tags: {e}")
            return []
    
    def _enhance_content_with_tags(self, content: str, related_tags: List[Dict[str, str]]) -> str:
        """Enhance content by appending related tag information."""
        if not related_tags:
            return content
        
        # Create tag enhancement section
        tag_section = "\n\n[相關標籤 Related Tags]:\n"
        for tag in related_tags:
            tag_section += f"- {tag['tag_zh']} ({tag['tag_en']})\n"
        
        return content + tag_section
    
    def save_database(self, db_path: str):
        """Save the structured vector database to disk."""
        db_data = {
            'config': self.config,
            'records': [asdict(record) for record in self.records],
            'created_at': datetime.now().isoformat(),
            'total_records': len(self.records)
        }
        
        with open(db_path, 'wb') as f:
            pickle.dump(db_data, f)
        
        # Save metadata
        metadata_path = Path(db_path).with_suffix('.metadata.json')
        metadata = {
            'database_type': 'structured_vector_db',
            'total_records': len(self.records),
            'vector_columns': list(self.vector_columns.keys()),
            'books': [book['book_name'] for book in self.config['content_structure']['books']],
            'created_at': datetime.now().isoformat()
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved structured vector database: {db_path}")
        logger.info(f"Saved metadata: {metadata_path}")
    
    def load_database(self, db_path: str) -> bool:
        """Load the structured vector database from disk."""
        try:
            with open(db_path, 'rb') as f:
                db_data = pickle.load(f)
            
            self.config = db_data['config']
            self.records = [VectorRecord(**record_data) for record_data in db_data['records']]
            
            logger.info(f"Loaded structured vector database with {len(self.records)} records")
            return True
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            return False
    
    def search_with_tags(self, query: str, vector_weights: Optional[Dict[str, float]] = None,
                        top_k: int = 5, use_tag_expansion: bool = True) -> List[Tuple[VectorRecord, float]]:
        """Enhanced search with tag-based query expansion and symptom matching."""
        
        # Note: Tag expansion now handled by tag_expansion_system.py
        expanded_query = query
        
        # Perform standard vector search with expanded query
        results = self.search_multi_vector(expanded_query, vector_weights, top_k)
        
        # Note: Tag-based scoring now integrated into embeddings during build phase
        
        return results
    
    def _enhance_results_with_tags(self, original_query: str, 
                                  results: List[Tuple[VectorRecord, float]]) -> List[Tuple[VectorRecord, float]]:
        """Enhance search results by boosting records that match TCM tags."""
        enhanced_results = []
        
        # Extract potential symptoms from query
        query_symptoms = self._extract_symptoms_from_query(original_query)
        
        for record, score in results:
            enhanced_score = score
            
            # Boost score if record content matches query symptoms
            if query_symptoms:
                content_text = f"{record.full_content} {record.symptoms_content}"
                symptom_matches = sum(1 for symptom in query_symptoms 
                                    if symptom in content_text)
                if symptom_matches > 0:
                    boost_factor = 1.0 + (symptom_matches * 0.1)  # 10% boost per matching symptom
                    enhanced_score *= boost_factor
                    logger.debug(f"Boosted score by {boost_factor:.2f} for {symptom_matches} symptom matches")
            
            enhanced_results.append((record, enhanced_score))
        
        # Re-sort by enhanced scores
        enhanced_results.sort(key=lambda x: x[1], reverse=True)
        return enhanced_results
    
    def _extract_symptoms_from_query(self, query: str) -> List[str]:
        """Extract known TCM symptoms from query text."""
        # Note: Symptom extraction now handled by embedded tags in records
        return []

    def search_multi_vector(self, query: str, vector_weights: Optional[Dict[str, float]] = None, 
                           top_k: int = 5) -> List[Tuple[VectorRecord, float]]:
        """Search using multiple vector columns with weighted scoring."""
        if not self.records:
            return []
        
        # Use default weights if not provided
        if vector_weights is None:
            vector_weights = self.retrieval_settings['vector_weights']
        
        # Generate query embeddings for each vector type
        query_embeddings = {}
        for vector_type in self.vector_columns.keys():
            if vector_weights.get(vector_type, 0) > 0:
                query_embeddings[vector_type] = self._get_embedding(query)
        
        # Calculate weighted similarity scores
        scored_records = []
        for record in self.records:
            total_score = 0.0
            valid_vectors = 0
            
            for vector_type, weight in vector_weights.items():
                if weight <= 0 or vector_type not in query_embeddings:
                    continue
                
                record_vector = getattr(record, vector_type)
                if not record_vector:
                    continue
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embeddings[vector_type], record_vector)
                total_score += similarity * weight
                valid_vectors += 1
            
            if valid_vectors > 0:
                avg_score = total_score / sum(w for w in vector_weights.values() if w > 0)
                scored_records.append((record, avg_score))
        
        # Sort by score and return top_k
        scored_records.sort(key=lambda x: x[1], reverse=True)
        return scored_records[:top_k]
    
    def get_adjacent_content(self, record: VectorRecord, expand_sections: int = 2, 
                           expand_pages: int = 5, expand_to_chapter: bool = False) -> List[VectorRecord]:
        """Get adjacent content around a specific record."""
        adjacent_records = []
        
        # Find records from the same book
        same_book_records = [r for r in self.records if r.book_id == record.book_id]
        same_book_records.sort(key=lambda x: (x.chapter_index, x.section_index))
        
        # Find the index of the current record
        try:
            current_index = same_book_records.index(record)
        except ValueError:
            return [record]  # Return just the original record if not found
        
        if expand_to_chapter:
            # Get entire chapter
            chapter_records = [r for r in same_book_records if r.chapter_index == record.chapter_index]
            adjacent_records.extend(chapter_records)
        else:
            # Get adjacent sections/pages
            start_idx = max(0, current_index - expand_sections)
            end_idx = min(len(same_book_records), current_index + expand_sections + 1)
            adjacent_records.extend(same_book_records[start_idx:end_idx])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_records = []
        for r in adjacent_records:
            record_id = (r.book_id, r.chapter_index, r.section_index)
            if record_id not in seen:
                seen.add(record_id)
                unique_records.append(r)
        
        return unique_records
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database."""
        if not self.records:
            return {'total_records': 0}
        
        books = list(set(r.book_name for r in self.records))
        chapters = list(set((r.book_id, r.chapter_index, r.chapter_title) for r in self.records))
        
        return {
            'total_records': len(self.records),
            'books': books,
            'total_chapters': len(chapters),
            'vector_columns': list(self.vector_columns.keys()),
            'avg_content_length': sum(r.char_count for r in self.records) / len(self.records)
        }
