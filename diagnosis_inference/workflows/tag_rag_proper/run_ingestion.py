#!/usr/bin/env python3
"""Simple script to run data ingestion for Tag RAG Proper system.

Run this once to build the vector database from the tagged TCM data.
Requires OPENAI_API_KEY environment variable to be set.
"""

import logging
import os
from database import setup_database
from ingestion import ingest_all_sections

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("Starting Tag RAG database ingestion...")
    print("=" * 50)
    
    # Setup database schema
    print("\n1. Setting up database schema...")
    setup_database()
    print("✓ Database schema ready")
    
    # Run ingestion
    print("\n2. Processing and ingesting sections...")
    count = ingest_all_sections(os.getenv('OPENAI_API_KEY'))
    
    print("\n" + "=" * 50)
    print(f"✅ Successfully ingested {count} sections into the database")
    print("\nYou can now use the query engine for diagnosis!")

if __name__ == "__main__":
    main()
