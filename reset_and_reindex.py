#!/usr/bin/env python3
"""
Reset database and indexes, then re-index documents with improved code
Run this to apply the latest improvements (OCR extraction, anti-hallucination, etc.)
"""

import os
import shutil

# Paths (adjust if needed)
base_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(base_dir, "rag_local.db")
indexes_dir = os.path.join(base_dir, "indexes")
images_dir = os.path.join(base_dir, "extracted_images")

print("="*60)
print("RESET AND RE-INDEX SCRIPT")
print("="*60)

# Delete database
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"✓ Deleted database: {db_path}")
else:
    print(f"  Database not found (already clean)")

# Delete indexes
if os.path.exists(indexes_dir):
    shutil.rmtree(indexes_dir)
    print(f"✓ Deleted indexes: {indexes_dir}")
else:
    print(f"  Indexes not found (already clean)")

# Delete extracted images
if os.path.exists(images_dir):
    shutil.rmtree(images_dir)
    print(f"✓ Deleted images: {images_dir}")
else:
    print(f"  Images not found (already clean)")

print("\n" + "="*60)
print("RESET COMPLETE!")
print("="*60)
print("\nNow run your notebook and re-index your documents.")
print("The new indexes will include:")
print("  • OCR text from images (searchable!)")
print("  • Improved anti-hallucination prompts")
print("  • Better image grouping")
print("="*60)
