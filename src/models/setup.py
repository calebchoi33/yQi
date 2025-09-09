#!/usr/bin/env python3
"""Setup script for yQi RAG system dependencies."""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_openai_api():
    """Check if OpenAI API key is available."""
    print("🔍 Checking OpenAI API configuration...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("✅ OpenAI API key found in environment variables")
        return True
    else:
        print("⚠️  OpenAI API key not found in environment variables")
        print("📋 Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   Or add it to your .env file in the evaluation directory")
        return False

def test_openai_connection():
    """Test OpenAI API connection."""
    try:
        import openai
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️  Cannot test API connection without API key")
            return False
            
        client = openai.OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="test"
        )
        print("✅ OpenAI API connection successful")
        return True
    except Exception as e:
        print(f"❌ OpenAI API connection failed: {e}")
        return False

def install_python_dependencies():
    """Install Python dependencies."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if requirements_file.exists():
        return run_command(
            f"{sys.executable} -m pip install -r {requirements_file}",
            "Installing Python dependencies"
        )
    else:
        # Install individual packages
        packages = [
            "openai>=1.0.0",
            "python-docx>=0.8.11", 
            "textract>=1.6.5",
            "numpy>=1.21.0"
        ]
        
        success = True
        for package in packages:
            if not run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}"):
                success = False
        return success

def check_documents():
    """Check if documents are available in the docs folder."""
    docs_path = Path(__file__).parent.parent / "docs"
    
    if not docs_path.exists():
        print(f"⚠️  Documents folder not found: {docs_path}")
        print("📁 Please create the docs folder and add your .doc/.docx files")
        return False
    
    doc_files = list(docs_path.glob("*.doc")) + list(docs_path.glob("*.docx")) + list(docs_path.glob("*.txt"))
    
    if not doc_files:
        print(f"⚠️  No documents found in: {docs_path}")
        print("📄 Please add .doc, .docx, or .txt files to the docs folder")
        return False
    
    print(f"✅ Found {len(doc_files)} document(s) in docs folder:")
    for doc in doc_files:
        print(f"   - {doc.name}")
    
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up yQi RAG System")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")
    
    # Install Python dependencies
    print("\n📦 Installing Python Dependencies")
    print("-" * 30)
    if not install_python_dependencies():
        print("❌ Failed to install some Python dependencies")
        print("💡 Try running: pip install -r models/requirements.txt")
    
    # Check OpenAI API
    print("\n🤖 Checking OpenAI API")
    print("-" * 20)
    if check_openai_api():
        print("\n🔗 Testing API Connection")
        print("-" * 25)
        test_openai_connection()
    else:
        print("⚠️  OpenAI API setup incomplete. RAG system may not work properly.")
    
    # Check documents
    print("\n📚 Checking Documents")
    print("-" * 20)
    check_documents()
    
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("Next steps:")
    print("1. Set your OpenAI API key in environment or .env file")
    print("2. Add your TCM documents to the docs/ folder")
    print("3. Run the evaluation platform: streamlit run evaluation/app.py")
    print("4. Select 'RAG (Local Documents)' in the model dropdown")

if __name__ == "__main__":
    main()
