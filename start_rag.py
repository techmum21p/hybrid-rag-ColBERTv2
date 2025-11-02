#!/usr/bin/env python3
"""
Simple launcher for Streamlit RAG Chatbot
Checks Ollama status and starts the app
"""

import sys
import time
import subprocess
import requests
from pathlib import Path

def print_header():
    """Print header"""
    print("\n" + "═" * 60)
    print("         Streamlit RAG Chatbot Launcher")
    print("═" * 60 + "\n")

def check_ollama():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_model(model_name):
    """Check if a specific model is available"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return model_name in result.stdout
    except:
        return False

def main():
    print_header()

    # Check Ollama
    print("🔍 Checking Ollama status...")
    if check_ollama():
        print("✅ Ollama is running\n")
    else:
        print("❌ Ollama is not running\n")
        print("Please start Ollama first:")
        print("   Option 1: Run 'ollama serve' in another terminal")
        print("   Option 2: Start Ollama app if you have it installed")
        print("\nThen run this script again.\n")
        sys.exit(1)

    # Check models
    print("🔍 Checking required models...")
    required_models = ["llama3.2:3b", "gemma3:4b"]
    missing_models = []

    for model in required_models:
        if check_model(model):
            print(f"   ✅ {model}")
        else:
            print(f"   ❌ {model} (missing)")
            missing_models.append(model)

    if missing_models:
        print("\n⚠️  Some models are missing. Install them with:")
        for model in missing_models:
            print(f"   ollama pull {model}")
        print()
        response = input("Continue anyway? (y/N): ").lower()
        if response != 'y':
            sys.exit(1)

    # Check for existing data
    print("\n🔍 Checking for existing data...")
    base_dir = Path(__file__).parent
    db_exists = (base_dir / "rag_local.db").exists()
    indexes_exist = (base_dir / "indexes" / "bm25s").exists() and \
                    (base_dir / "indexes" / "colbert").exists()

    if db_exists:
        print("   ✅ Database found")
    else:
        print("   ℹ️  No database found (will be created)")

    if indexes_exist:
        print("   ✅ Indexes found")
    else:
        print("   ℹ️  No indexes found (upload documents to create)")

    # Show status
    print("\n" + "═" * 60)
    if db_exists and indexes_exist:
        print("📚 Existing data found - chatbot will auto-initialize")
    elif db_exists:
        print("📚 Database found but no indexes - re-upload documents")
    else:
        print("📚 No data found - upload documents to get started")
    print("═" * 60 + "\n")

    # Start Streamlit
    print("🚀 Starting Streamlit app...")
    print("\nThe app will open in your browser at: http://localhost:8501")
    print("\nPress Ctrl+C to stop the server\n")
    print("═" * 60 + "\n")

    try:
        subprocess.run(["streamlit", "run", "streamlit_rag.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        print("✅ Cleanup complete. Goodbye!\n")

if __name__ == "__main__":
    main()
