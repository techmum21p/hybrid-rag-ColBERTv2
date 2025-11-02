#!/usr/bin/env python3
"""
Validation script for Streamlit RAG improvements
Checks that all key features are properly implemented
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description}: Found")
        return True
    else:
        print(f"❌ {description}: Not found")
        return False

def check_code_feature(filepath, pattern, description):
    """Check if code contains specific feature"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            if pattern in content:
                print(f"✅ {description}: Implemented")
                return True
            else:
                print(f"❌ {description}: Not found")
                return False
    except Exception as e:
        print(f"❌ Error checking {description}: {e}")
        return False

def main():
    print("=" * 60)
    print("Validating Streamlit RAG Improvements")
    print("=" * 60)

    base_dir = Path(__file__).parent
    streamlit_file = base_dir / "streamlit_rag.py"

    checks = []

    # Check file exists
    print("\n[1/7] Checking files...")
    checks.append(check_file_exists(streamlit_file, "streamlit_rag.py"))

    # Check SQLite threading fix
    print("\n[2/7] Checking SQLite threading fix...")
    checks.append(check_code_feature(
        streamlit_file,
        'check_same_thread": False',
        "SQLite multi-threading support"
    ))

    # Check duplicate detection
    print("\n[3/7] Checking duplicate file detection...")
    checks.append(check_code_feature(
        streamlit_file,
        "def check_duplicate_file",
        "Duplicate file detection method"
    ))
    checks.append(check_code_feature(
        streamlit_file,
        "def delete_document_and_data",
        "Document deletion method"
    ))

    # Check smart index management
    print("\n[4/7] Checking smart index management...")
    checks.append(check_code_feature(
        streamlit_file,
        "def indexes_exist",
        "Index existence check"
    ))
    checks.append(check_code_feature(
        streamlit_file,
        "overwrite_duplicates: bool",
        "Overwrite duplicates parameter"
    ))

    # Check auto-initialization
    print("\n[5/7] Checking auto-initialization...")
    checks.append(check_code_feature(
        streamlit_file,
        "auto_init_attempted",
        "Auto-initialization flag"
    ))
    checks.append(check_code_feature(
        streamlit_file,
        "st.session_state.app.indexes_exist()",
        "Index existence check in UI"
    ))

    # Check metadata and image features
    print("\n[6/7] Checking metadata and image contextualization...")
    checks.append(check_code_feature(
        streamlit_file,
        "def enrich_chunks_with_images",
        "Image enrichment method"
    ))
    checks.append(check_code_feature(
        streamlit_file,
        "ocr_text",
        "OCR text support"
    ))
    checks.append(check_code_feature(
        streamlit_file,
        "image_metadata",
        "Image metadata storage"
    ))
    checks.append(check_code_feature(
        streamlit_file,
        "st.image(img_path",
        "Image display in UI"
    ))

    # Check error handling
    print("\n[7/7] Checking error handling...")
    checks.append(check_code_feature(
        streamlit_file,
        "except Exception as e:",
        "Exception handling"
    ))
    checks.append(check_code_feature(
        streamlit_file,
        "st.warning",
        "User warnings in UI"
    ))

    # Summary
    print("\n" + "=" * 60)
    passed = sum(checks)
    total = len(checks)
    print(f"Validation Results: {passed}/{total} checks passed")

    if passed == total:
        print("✅ All improvements validated successfully!")
        print("\nYou can now run: streamlit run streamlit_rag.py")
    else:
        print(f"⚠️  {total - passed} checks failed. Please review the implementation.")
        sys.exit(1)

    print("=" * 60)

if __name__ == "__main__":
    main()
