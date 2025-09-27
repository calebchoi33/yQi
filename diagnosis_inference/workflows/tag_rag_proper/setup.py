#!/usr/bin/env python3
# Use system Python 3.9+ for better compatibility
"""Setup script for Tag RAG Proper system."""

import os
import sys
import subprocess
from pathlib import Path
from database import setup_database
from ingestion import ingest_all_sections

def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✓ Requirements installed successfully")
    return True

def check_api_key():
    """Check if OpenAI API key is set."""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("✓ OpenAI API key found")
        return True
    else:
        print("✗ OpenAI API key not found")
        print("Please set your API key: export OPENAI_API_KEY='your-key-here'")
        return False

def check_data_file():
    """Check if tags.json file exists."""
    tags_path = Path("《人紀傷寒論》_tags.json")
    if tags_path.exists():
        print("✓ Tags data file found")
        return True
    else:
        print(f"✗ Tags data file not found at {tags_path}")
        return False

def run_ingestion():
    """Run the data ingestion process."""
    print("\n4. Running data ingestion...")
    setup_database()  # Setup PostgreSQL database and tables
    api_key = os.getenv('OPENAI_API_KEY')
    count = ingest_all_sections(api_key)
    print(f"✓ Successfully ingested {count} sections")
    return True

def main():
    """Main setup function."""
    print("Tag RAG Proper System Setup")
    print("=" * 40)
    
    # Check prerequisites
    checks = [
        ("Installing requirements", install_requirements),
        ("Checking API key", check_api_key),
        ("Checking data file", check_data_file)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n{check_name}...")
        if not check_func():
            all_passed = False
    
    if not all_passed:
        print("\n⚠️  Some checks failed. Please resolve the issues above before proceeding.")
        return
    
    # Run ingestion
    print(f"\n{'='*40}")
    run_ingestion()
    print("\n🎉 Setup completed successfully!")
    print("\n=== Usage Examples ===")
    print("\n1. Basic query:")
    print("   from query_engine import query")
    print("   results = query('fever and headache', 'symptoms', k=5, api_key=api_key)")
    print("\n2. Multi-family query:")
    print("   from query_engine import multi_family_query")
    print("   multi_results = multi_family_query('floating pulse with fever', k=3, api_key=api_key)")
    print("\n3. Run tests:")
    print("   python test_system.py")

if __name__ == "__main__":
    main()
