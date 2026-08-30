#!/usr/bin/env python3
"""
Fix Invalid UTF-8 in Database Chunks
=====================================
This script cleans up any chunks in the database that contain invalid UTF-8 characters.
"""

import os
import sys
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# Database Models
class Base(DeclarativeBase):
    pass

class Chunk(Base):
    __tablename__ = 'chunks'

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    heading_path = Column(String(500))
    token_count = Column(Integer)
    has_images = Column(Boolean, default=False)
    chunk_metadata = Column(Text)

def sanitize_utf8(text: str) -> str:
    """Robust UTF-8 sanitization"""
    if not text:
        return text

    try:
        # Try to encode/decode with error handling
        # This removes invalid UTF-8 sequences
        clean_text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')

        # Also remove null bytes which can cause issues
        clean_text = clean_text.replace('\x00', '')

        # Remove other problematic control characters but keep newlines and tabs
        clean_text = ''.join(char for char in clean_text
                            if char == '\n' or char == '\t' or char == '\r' or ord(char) >= 32 or ord(char) == 9)

        return clean_text
    except Exception as e:
        print(f"    ⚠️  Sanitization error: {e}")
        # Last resort: keep only ASCII
        return text.encode('ascii', errors='ignore').decode('ascii', errors='ignore')

def fix_database_utf8(db_path: str):
    """Fix all chunks in the database with invalid UTF-8"""

    print("\n" + "="*60)
    print("Database UTF-8 Cleanup Tool")
    print("="*60)
    print(f"\nDatabase: {db_path}\n")

    # Connect to database
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Get all chunks
    chunks = session.query(Chunk).all()
    total_chunks = len(chunks)

    print(f"Found {total_chunks} chunks to check...\n")

    fixed_count = 0

    for i, chunk in enumerate(chunks, 1):
        try:
            # Try to encode the text - if it fails, it has invalid UTF-8
            chunk.text.encode('utf-8')

            # Also check metadata
            if chunk.chunk_metadata:
                chunk.chunk_metadata.encode('utf-8')

            if chunk.heading_path:
                chunk.heading_path.encode('utf-8')

        except UnicodeEncodeError:
            print(f"  Chunk {i}/{total_chunks} (ID: {chunk.id}): Found invalid UTF-8, fixing...")

            # Sanitize all text fields
            chunk.text = sanitize_utf8(chunk.text)

            if chunk.chunk_metadata:
                chunk.chunk_metadata = sanitize_utf8(chunk.chunk_metadata)

            if chunk.heading_path:
                chunk.heading_path = sanitize_utf8(chunk.heading_path)

            fixed_count += 1

        # Show progress every 100 chunks
        if i % 100 == 0:
            print(f"  Checked {i}/{total_chunks} chunks...")

    # Commit changes
    if fixed_count > 0:
        print(f"\nCommitting changes...")
        session.commit()
        print(f"✅ Fixed {fixed_count} chunks with invalid UTF-8")
    else:
        print(f"✅ No invalid UTF-8 found - database is clean!")

    session.close()
    print("\n" + "="*60)
    print("Cleanup complete!")
    print("="*60)

if __name__ == "__main__":
    # Get database path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "rag_local.db")

    if not os.path.exists(db_path):
        print(f"❌ Database not found at: {db_path}")
        sys.exit(1)

    # Run cleanup
    fix_database_utf8(db_path)

    print("\n💡 Next steps:")
    print("   1. Restart your Jupyter kernel")
    print("   2. Re-run your notebook cells")
    print("   3. Try your queries again")
    print("\n   The invalid UTF-8 should no longer cause issues!")
