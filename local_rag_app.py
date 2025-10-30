"""
Local RAG Application for Mac Mini M4 with Ollama
==================================================

Features:
- Runs entirely on Mac Mini M4 (no cloud dependencies)
- PDF processing with MarkItDown
- Markdown-aware semantic chunking
- BM25s + ColBERTv2 hybrid retrieval with RRF fusion
- ColBERTv2 reranking
- Ollama for local LLM generation
- Detailed timing metrics for all operations
- Local SQLite database

Optimized for Apple Silicon M4
"""

import os
import json
import time
import sqlite3
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import hashlib
import re

# Core libraries
import bm25s
import numpy as np
from ragatouille import RAGPretrainedModel

# Document processing
import pymupdf4llm
import fitz  # PyMuPDF
from transformers import AutoTokenizer
from PIL import Image
import io

# Ollama integration
import requests


# ============================================================================
# CONFIGURATION
# ============================================================================

class LocalRAGConfig:
    """Configuration for local RAG system on Mac Mini M4"""
    
    # Paths
    BASE_DIR = Path("./rag_local")
    DB_PATH = BASE_DIR / "rag_local.db"
    INDEXES_DIR = BASE_DIR / "indexes"
    BM25_INDEX_PATH = INDEXES_DIR / "bm25s"
    COLBERT_INDEX_PATH = INDEXES_DIR / "colbert"
    
    # Markdown-aware chunking
    MIN_CHUNK_SIZE = 256
    MAX_CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 128
    
    # Retrieval
    BM25_TOP_K = 100
    COLBERT_TOP_K = 100
    FINAL_TOP_K = 10
    
    # Ollama settings
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "gemma2:2b"  # Gemma2 2B for text
    OLLAMA_VISION_MODEL = "gemma2:27b-vision"  # Gemma2 27B Vision for images
    OLLAMA_TIMEOUT = 120
    
    # Models
    COLBERT_MODEL = "colbert-ir/colbertv2.0"
    TOKENIZER_MODEL = "bert-base-uncased"
    
    def __init__(self):
        # Create directories
        self.BASE_DIR.mkdir(exist_ok=True)
        self.INDEXES_DIR.mkdir(parents=True, exist_ok=True)
        self.BM25_INDEX_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================================
# TIMING UTILITY
# ============================================================================

class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str, verbose: bool = True):
        self.operation_name = operation_name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.verbose:
            print(f"⏱️  {self.operation_name}...", flush=True)
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            print(f"   ✓ Completed in {self.elapsed:.2f}s")
    
    def get_elapsed(self) -> float:
        return self.elapsed if self.elapsed else 0.0


# ============================================================================
# DATABASE MANAGER
# ============================================================================

class LocalDatabase:
    """SQLite database for local storage"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database and create tables"""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        cursor = self.conn.cursor()
        
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_hash TEXT UNIQUE NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'processing'
            )
        """)
        
        # Chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                heading_path TEXT,
                token_count INTEGER,
                metadata TEXT,
                has_images INTEGER DEFAULT 0,
                image_paths TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
        
        self.conn.commit()
    
    def add_document(self, filename: str, file_hash: str) -> int:
        """Add a document to the database"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO documents (filename, file_hash, status) VALUES (?, ?, 'processing')",
            (filename, file_hash)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def add_chunk(self, document_id: int, chunk_index: int, text: str, 
                  heading_path: str = "", token_count: int = 0, metadata: dict = None,
                  has_images: bool = False, image_paths: list = None):
        """Add a chunk to the database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO chunks (document_id, chunk_index, text, heading_path, token_count, 
                              metadata, has_images, image_paths)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (document_id, chunk_index, text, heading_path, token_count, 
              json.dumps(metadata) if metadata else "{}",
              1 if has_images else 0,
              json.dumps(image_paths) if image_paths else "[]"))
        self.conn.commit()
    
    def get_chunk(self, chunk_id: int) -> Optional[Dict]:
        """Get a chunk by ID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def get_all_chunks(self) -> List[Dict]:
        """Get all chunks"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM chunks ORDER BY document_id, chunk_index")
        return [dict(row) for row in cursor.fetchall()]
    
    def update_document_status(self, document_id: int, status: str):
        """Update document status"""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE documents SET status = ? WHERE id = ?",
            (status, document_id)
        )
        self.conn.commit()
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as count FROM documents")
        doc_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) as count FROM chunks")
        chunk_count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT AVG(token_count) as avg_tokens 
            FROM chunks WHERE token_count > 0
        """)
        avg_tokens = cursor.fetchone()[0] or 0
        
        return {
            'documents': doc_count,
            'chunks': chunk_count,
            'avg_chunk_size': int(avg_tokens)
        }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


# ============================================================================
# MARKDOWN-AWARE SEMANTIC CHUNKER
# ============================================================================

class MarkdownSemanticChunker:
    """Markdown-aware semantic chunking"""
    
    def __init__(self, config: LocalRAGConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_MODEL)
    
    def chunk_markdown(self, markdown_text: str, doc_context: str = "") -> List[Dict]:
        """Create semantically meaningful chunks"""
        sections = self._parse_markdown_hierarchy(markdown_text)
        chunks = self._create_chunks_from_sections(sections, doc_context)
        optimized_chunks = self._optimize_chunks(chunks)
        return optimized_chunks
    
    def _parse_markdown_hierarchy(self, text: str) -> List[Dict]:
        """Parse markdown into hierarchical sections"""
        lines = text.split('\n')
        sections = []
        current_section = None
        heading_stack = []
        
        for line in lines:
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if heading_match:
                if current_section:
                    sections.append(current_section)
                
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                heading_stack = [(lvl, ttl) for lvl, ttl in heading_stack if lvl < level]
                heading_stack.append((level, title))
                
                parent_path = ' > '.join([ttl for _, ttl in heading_stack[:-1]])
                full_path = ' > '.join([ttl for _, ttl in heading_stack])
                
                current_section = {
                    'level': level,
                    'title': title,
                    'content': '',
                    'parent_path': parent_path,
                    'full_path': full_path
                }
            else:
                if current_section is not None:
                    current_section['content'] += line + '\n'
                else:
                    if not sections or sections[-1]['level'] != 0:
                        sections.append({
                            'level': 0,
                            'title': 'Introduction',
                            'content': line + '\n',
                            'parent_path': '',
                            'full_path': 'Introduction'
                        })
                    else:
                        sections[-1]['content'] += line + '\n'
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _create_chunks_from_sections(self, sections: List[Dict], doc_context: str) -> List[Dict]:
        """Create chunks from sections"""
        chunks = []
        current_chunk = None
        
        for section in sections:
            section_text = self._format_section_text(section)
            section_tokens = self._count_tokens(section_text)
            
            if section_tokens > self.config.MAX_CHUNK_SIZE:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = None
                split_chunks = self._split_large_section(section, doc_context)
                chunks.extend(split_chunks)
            
            elif section_tokens >= self.config.MIN_CHUNK_SIZE:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = None
                chunks.append({
                    'text': section_text,
                    'heading_path': section['full_path'],
                    'level': section['level'],
                    'token_count': section_tokens,
                    'doc_context': doc_context,
                    'type': 'section'
                })
            
            else:
                if current_chunk is None:
                    current_chunk = {
                        'text': section_text,
                        'heading_path': section['parent_path'] or section['title'],
                        'level': section['level'],
                        'token_count': section_tokens,
                        'doc_context': doc_context,
                        'type': 'accumulated',
                        'sections': [section['title']]
                    }
                else:
                    combined_text = current_chunk['text'] + '\n\n' + section_text
                    combined_tokens = self._count_tokens(combined_text)
                    
                    if combined_tokens <= self.config.MAX_CHUNK_SIZE:
                        current_chunk['text'] = combined_text
                        current_chunk['token_count'] = combined_tokens
                        current_chunk['sections'].append(section['title'])
                    else:
                        chunks.append(current_chunk)
                        current_chunk = {
                            'text': section_text,
                            'heading_path': section['parent_path'] or section['title'],
                            'level': section['level'],
                            'token_count': section_tokens,
                            'doc_context': doc_context,
                            'type': 'accumulated',
                            'sections': [section['title']]
                        }
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_large_section(self, section: Dict, doc_context: str) -> List[Dict]:
        """Split large section at paragraph boundaries"""
        heading_text = f"# {section['title']}\n\n"
        parent_context = f"Context: {section['parent_path']}\n\n" if section['parent_path'] else ""
        
        paragraphs = re.split(r'\n\n+', section['content'].strip())
        
        chunks = []
        current_text = heading_text + parent_context
        current_tokens = self._count_tokens(current_text)
        
        for para in paragraphs:
            para_tokens = self._count_tokens(para)
            
            if current_tokens + para_tokens <= self.config.MAX_CHUNK_SIZE:
                current_text += para + '\n\n'
                current_tokens += para_tokens
            else:
                if current_text.strip() != heading_text.strip():
                    chunks.append({
                        'text': current_text.strip(),
                        'heading_path': section['full_path'],
                        'level': section['level'],
                        'token_count': current_tokens,
                        'doc_context': doc_context,
                        'type': 'split_section',
                        'part': len(chunks) + 1
                    })
                
                current_text = heading_text + parent_context + para + '\n\n'
                current_tokens = self._count_tokens(current_text)
        
        if current_text.strip():
            chunks.append({
                'text': current_text.strip(),
                'heading_path': section['full_path'],
                'level': section['level'],
                'token_count': current_tokens,
                'doc_context': doc_context,
                'type': 'split_section',
                'part': len(chunks) + 1
            })
        
        return chunks
    
    def _optimize_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Optimize chunks by merging small ones"""
        optimized = []
        i = 0
        
        while i < len(chunks):
            chunk = chunks[i]
            
            if (chunk['token_count'] < self.config.MIN_CHUNK_SIZE and 
                i < len(chunks) - 1):
                
                next_chunk = chunks[i + 1]
                combined_text = chunk['text'] + '\n\n' + next_chunk['text']
                combined_tokens = self._count_tokens(combined_text)
                
                if combined_tokens <= self.config.MAX_CHUNK_SIZE:
                    merged_chunk = {
                        'text': combined_text,
                        'heading_path': chunk['heading_path'],
                        'token_count': combined_tokens,
                        'doc_context': chunk['doc_context'],
                        'type': 'merged'
                    }
                    optimized.append(merged_chunk)
                    i += 2
                    continue
            
            optimized.append(chunk)
            i += 1
        
        return optimized
    
    def _format_section_text(self, section: Dict) -> str:
        """Format section with heading and context"""
        parts = []
        
        if section['parent_path']:
            parts.append(f"[Context: {section['parent_path']}]")
        
        if section['title'] and section['title'] != 'Introduction':
            heading_prefix = '#' * section['level']
            parts.append(f"{heading_prefix} {section['title']}")
        
        parts.append(section['content'].strip())
        
        return '\n\n'.join(parts)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))


# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    """Process PDFs with PyMuPDF4LLM and extract images"""
    
    def __init__(self, config: LocalRAGConfig):
        self.config = config
        self.chunker = MarkdownSemanticChunker(config)
        
        # Create images directory
        self.images_dir = config.BASE_DIR / "images"
        self.images_dir.mkdir(exist_ok=True)
    
    def process_pdf(self, pdf_path: str, db: LocalDatabase) -> Tuple[List[Dict], int]:
        """Process PDF and return chunks with timing"""
        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(pdf_path)}")
        print(f"{'='*70}\n")
        
        # Calculate file hash
        with open(pdf_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Create document record
        doc_id = db.add_document(os.path.basename(pdf_path), file_hash)
        
        # Step 1: Convert PDF to Markdown with PyMuPDF4LLM
        with Timer("[Step 1/4] Converting PDF to Markdown (PyMuPDF4LLM)") as t1:
            markdown_text = pymupdf4llm.to_markdown(pdf_path)
            print(f"   • Extracted {len(markdown_text):,} characters")
        
        # Step 2: Extract images from PDF
        with Timer("[Step 2/4] Extracting images from PDF") as t2:
            images = self._extract_images(pdf_path, doc_id)
            print(f"   • Extracted {len(images)} images")
        
        # Step 3: Chunk markdown
        with Timer("[Step 3/4] Markdown-aware semantic chunking") as t3:
            doc_context = f"Document: {os.path.basename(pdf_path)}\n\n{markdown_text[:500]}"
            chunks = self.chunker.chunk_markdown(markdown_text, doc_context)
            avg_size = sum(c['token_count'] for c in chunks) // len(chunks) if chunks else 0
            print(f"   • Created {len(chunks)} semantic chunks")
            print(f"   • Average chunk size: {avg_size} tokens")
        
        # Step 4: Save to database with image associations
        with Timer("[Step 4/4] Saving chunks to database") as t4:
            for idx, chunk in enumerate(chunks):
                # Find images that belong to this chunk (based on text proximity)
                chunk_images = self._associate_images_to_chunk(chunk, images)
                
                db.add_chunk(
                    document_id=doc_id,
                    chunk_index=idx,
                    text=chunk['text'],
                    heading_path=chunk.get('heading_path', ''),
                    token_count=chunk.get('token_count', 0),
                    metadata={k: v for k, v in chunk.items() if k not in ['text']},
                    has_images=len(chunk_images) > 0,
                    image_paths=chunk_images
                )
            db.update_document_status(doc_id, 'indexed')
        
        total_time = t1.get_elapsed() + t2.get_elapsed() + t3.get_elapsed() + t4.get_elapsed()
        print(f"\n✅ Total processing time: {total_time:.2f}s\n")
        
        return chunks, doc_id
    
    def _extract_images(self, pdf_path: str, doc_id: int) -> List[Dict]:
        """Extract images from PDF using PyMuPDF"""
        images = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Save image
                image_filename = f"doc{doc_id}_page{page_num}_img{img_idx}.{image_ext}"
                image_path = self.images_dir / image_filename
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                images.append({
                    'path': str(image_path),
                    'page': page_num,
                    'index': img_idx,
                    'filename': image_filename
                })
        
        doc.close()
        return images
    
    def _associate_images_to_chunk(self, chunk: Dict, images: List[Dict]) -> List[str]:
        """Associate images to chunks based on page proximity"""
        # Simple heuristic: associate images based on heading or page reference in text
        # For now, return empty list - can be enhanced with more sophisticated logic
        return []


# ============================================================================
# DUAL INDEXER (BM25s + ColBERTv2)
# ============================================================================

class DualIndexer:
    """Manages BM25s and ColBERTv2 indexes"""
    
    def __init__(self, config: LocalRAGConfig):
        self.config = config
        self.bm25_retriever = None
        self.colbert_model = None
    
    def build_indexes(self, chunks: List[Dict], index_name: str = "rag_index"):
        """Build both indexes with timing"""
        print(f"\n{'='*70}")
        print("Building Indexes")
        print(f"{'='*70}\n")
        
        # BM25s index
        with Timer("[BM25s] Building lexical search index"):
            corpus = [chunk['text'] for chunk in chunks]
            corpus_tokens = bm25s.tokenize(
                corpus,
                stopwords="en",
                stemmer=bm25s.stemmer.Stemmer.Stemmer("english")
            )
            self.bm25_retriever = bm25s.BM25()
            self.bm25_retriever.index(corpus_tokens)
            self.bm25_retriever.save(str(self.config.BM25_INDEX_PATH))
            print(f"   • Indexed {len(chunks)} chunks")
        
        # ColBERTv2 index
        with Timer("[ColBERT] Building semantic search index"):
            self.colbert_model = RAGPretrainedModel.from_pretrained(
                self.config.COLBERT_MODEL
            )
            self.colbert_model.index(
                collection=corpus,
                index_name=index_name,
                max_document_length=512,
                split_documents=False
            )
            print(f"   • Indexed {len(chunks)} chunks")
        
        print(f"\n✅ Indexes built successfully!\n")
    
    def load_indexes(self, index_name: str = "rag_index"):
        """Load existing indexes"""
        print("Loading indexes...")
        
        # Load BM25s
        self.bm25_retriever = bm25s.BM25.load(str(self.config.BM25_INDEX_PATH))
        
        # Load ColBERT
        colbert_path = f".ragatouille/colbert/indexes/{index_name}"
        if not Path(colbert_path).exists():
            raise FileNotFoundError(f"ColBERT index not found: {colbert_path}")
        
        self.colbert_model = RAGPretrainedModel.from_index(colbert_path)
        
        print("✅ Indexes loaded!")


# ============================================================================
# HYBRID RETRIEVER WITH TIMING
# ============================================================================

class HybridRetriever:
    """Three-stage retrieval with detailed timing"""
    
    def __init__(self, config: LocalRAGConfig, indexer: DualIndexer, db: LocalDatabase):
        self.config = config
        self.indexer = indexer
        self.db = db
    
    def retrieve(self, query: str, top_k: int = None) -> Tuple[List[Dict], Dict]:
        """Retrieve with detailed timing breakdown"""
        if top_k is None:
            top_k = self.config.FINAL_TOP_K
        
        timings = {}
        
        # Stage 1: BM25s
        with Timer("BM25s retrieval", verbose=False) as t1:
            bm25_results = self._bm25_search(query, k=self.config.BM25_TOP_K)
        timings['bm25'] = t1.get_elapsed()
        
        # Stage 2: ColBERT
        with Timer("ColBERT retrieval", verbose=False) as t2:
            colbert_results = self._colbert_search(query, k=self.config.COLBERT_TOP_K)
        timings['colbert'] = t2.get_elapsed()
        
        # Fusion
        with Timer("RRF fusion", verbose=False) as t3:
            fused_results = self._reciprocal_rank_fusion(bm25_results, colbert_results)
            candidates = fused_results[:50]
        timings['fusion'] = t3.get_elapsed()
        
        # Fetch chunks
        with Timer("Fetching chunks", verbose=False) as t4:
            candidate_chunks = self._fetch_chunks_from_db([r['chunk_id'] for r in candidates])
        timings['fetch'] = t4.get_elapsed()
        
        # Reranking
        with Timer("ColBERT reranking", verbose=False) as t5:
            reranked = self._colbert_rerank(query, candidate_chunks, top_k=top_k)
        timings['rerank'] = t5.get_elapsed()
        
        timings['total'] = sum(timings.values())
        
        return reranked, timings
    
    def _bm25_search(self, query: str, k: int) -> List[Dict]:
        """BM25s search"""
        query_tokens = bm25s.tokenize(
            query,
            stopwords="en",
            stemmer=bm25s.stemmer.Stemmer.Stemmer("english")
        )
        results, scores = self.indexer.bm25_retriever.retrieve(query_tokens, k=k)
        return [
            {'chunk_id': int(results[0][i]), 'score': float(scores[0][i]), 'source': 'bm25'}
            for i in range(len(results[0]))
        ]
    
    def _colbert_search(self, query: str, k: int) -> List[Dict]:
        """ColBERT search"""
        results = self.indexer.colbert_model.search(query=query, k=k)
        return [
            {'chunk_id': r['document_id'], 'score': r['score'], 'source': 'colbert'}
            for r in results
        ]
    
    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict],
        colbert_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """RRF fusion"""
        scores = {}
        
        for rank, result in enumerate(bm25_results, 1):
            chunk_id = result['chunk_id']
            scores[chunk_id] = scores.get(chunk_id, 0) + (1 / (k + rank))
        
        for rank, result in enumerate(colbert_results, 1):
            chunk_id = result['chunk_id']
            scores[chunk_id] = scores.get(chunk_id, 0) + (1 / (k + rank))
        
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{'chunk_id': cid, 'rrf_score': score} for cid, score in sorted_results]
    
    def _fetch_chunks_from_db(self, chunk_ids: List[int]) -> List[Dict]:
        """Fetch chunks from database"""
        chunks = []
        for chunk_id in chunk_ids:
            chunk = self.db.get_chunk(chunk_id)
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def _colbert_rerank(self, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
        """ColBERT reranking"""
        if not chunks:
            return []
        
        documents = [chunk['text'] for chunk in chunks]
        reranked_results = self.indexer.colbert_model.rerank(
            query=query,
            documents=documents,
            k=top_k
        )
        
        final_results = []
        for result in reranked_results:
            original_chunk = chunks[result['result_index']]
            final_results.append({
                **original_chunk,
                'score': result['score'],
                'rank': result['rank']
            })
        
        return final_results


# ============================================================================
# OLLAMA CLIENT
# ============================================================================

class OllamaClient:
    """Client for Ollama API with vision support"""
    
    def __init__(self, config: LocalRAGConfig):
        self.config = config
        self.base_url = config.OLLAMA_BASE_URL
        self.model = config.OLLAMA_MODEL
        self.vision_model = config.OLLAMA_VISION_MODEL
    
    def generate(self, prompt: str, system_prompt: str = "", images: List[str] = None) -> str:
        """Generate response from Ollama (with optional images for vision model)"""
        url = f"{self.base_url}/api/generate"
        
        # Use vision model if images are provided
        model = self.vision_model if images else self.model
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        # Add images for vision model
        if images:
            payload["images"] = []
            for image_path in images:
                try:
                    # Read and encode image as base64
                    import base64
                    with open(image_path, 'rb') as img_file:
                        image_data = base64.b64encode(img_file.read()).decode('utf-8')
                        payload["images"].append(image_data)
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not load image {image_path}: {e}")
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.OLLAMA_TIMEOUT
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '')
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Error calling Ollama: {e}")
            return f"Error: Could not generate response. Make sure Ollama is running with: ollama serve"
    
    def check_connection(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


# ============================================================================
# RAG CHATBOT
# ============================================================================

class RAGChatbot:
    """RAG chatbot with Ollama"""
    
    def __init__(self, config: LocalRAGConfig, retriever: HybridRetriever):
        self.config = config
        self.retriever = retriever
        self.ollama = OllamaClient(config)
        self.conversation_history = []
    
    def chat(self, query: str) -> Dict:
        """Chat with timing breakdown"""
        # Retrieve chunks
        print(f"\n🔍 Retrieving relevant chunks...")
        retrieved_chunks, retrieval_timings = self.retriever.retrieve(query)
        
        # Print retrieval timing
        print(f"   • BM25s: {retrieval_timings['bm25']:.3f}s")
        print(f"   • ColBERT: {retrieval_timings['colbert']:.3f}s")
        print(f"   • Fusion: {retrieval_timings['fusion']:.3f}s")
        print(f"   • Fetch: {retrieval_timings['fetch']:.3f}s")
        print(f"   • Rerank: {retrieval_timings['rerank']:.3f}s")
        print(f"   ✓ Total retrieval: {retrieval_timings['total']:.3f}s")
        print(f"   • Retrieved {len(retrieved_chunks)} chunks\n")
        
        # Build context
        context = self._build_context(retrieved_chunks)
        
        # Generate response
        print(f"🤖 Generating response with Ollama...")
        with Timer("Response generation", verbose=False) as gen_timer:
            response = self._generate_response(query, context, retrieved_chunks)
        
        print(f"   ✓ Generated in {gen_timer.get_elapsed():.3f}s\n")
        
        # Total time
        total_time = retrieval_timings['total'] + gen_timer.get_elapsed()
        print(f"⏱️  Total query time: {total_time:.3f}s")
        print(f"   • Retrieval: {retrieval_timings['total']:.3f}s ({retrieval_timings['total']/total_time*100:.1f}%)")
        print(f"   • Generation: {gen_timer.get_elapsed():.3f}s ({gen_timer.get_elapsed()/total_time*100:.1f}%)")
        
        return {
            'response': response,
            'sources': self._format_sources(retrieved_chunks),
            'retrieved_chunks': len(retrieved_chunks),
            'timings': {
                'retrieval': retrieval_timings,
                'generation': gen_timer.get_elapsed(),
                'total': total_time
            }
        }
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context from chunks"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            heading = chunk.get('heading_path', '')
            heading_str = f" ({heading})" if heading else ""
            context_parts.append(f"[Source {i}{heading_str}]\n{chunk['text']}\n")
        return "\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str, chunks: List[Dict]) -> str:
        """Generate response with Ollama (uses vision model if images present)"""
        system_prompt = "You are a helpful assistant. Answer questions based on the provided context. Cite sources using [Source N] notation."
        
        # Collect images from chunks
        images_to_analyze = []
        for chunk in chunks[:3]:  # Analyze images from top 3 chunks
            if chunk.get('has_images') and chunk.get('image_paths'):
                try:
                    image_paths = json.loads(chunk['image_paths']) if isinstance(chunk['image_paths'], str) else chunk['image_paths']
                    images_to_analyze.extend(image_paths)
                except:
                    pass
        
        prompt = f"""Context from documents:

{context}

Question: {query}

Answer the question based on the context provided. If the context doesn't contain enough information, say so. Always cite your sources using [Source N] notation."""
        
        if images_to_analyze:
            prompt += f"\n\nNote: This query includes {len(images_to_analyze)} image(s) from the document. Analyze them if relevant to the question."
        
        return self.ollama.generate(prompt, system_prompt, images=images_to_analyze if images_to_analyze else None)
    
    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Format sources"""
        return [
            {
                'source_id': i + 1,
                'chunk_id': chunk['id'],
                'document_id': chunk['document_id'],
                'heading': chunk.get('heading_path', ''),
                'score': chunk.get('score', 0),
                'preview': chunk['text'][:200] + "..."
            }
            for i, chunk in enumerate(chunks)
        ]


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class LocalRAGApplication:
    """Main application orchestrator"""
    
    def __init__(self, config: LocalRAGConfig = None):
        self.config = config or LocalRAGConfig()
        self.db = LocalDatabase(self.config.DB_PATH)
        self.processor = DocumentProcessor(self.config)
        self.indexer = DualIndexer(self.config)
        self.retriever = None
        self.chatbot = None
    
    def index_document(self, pdf_path: str):
        """Index a single PDF"""
        # Process document
        chunks, doc_id = self.processor.process_pdf(pdf_path, self.db)
        
        # Get all chunks from database
        all_chunks = self.db.get_all_chunks()
        
        # Build indexes
        self.indexer.build_indexes(all_chunks)
        
        print(f"✅ Document indexed successfully!")
        print(f"   • Total documents: {self.db.get_stats()['documents']}")
        print(f"   • Total chunks: {self.db.get_stats()['chunks']}")
    
    def initialize_chatbot(self):
        """Initialize chatbot with existing indexes"""
        # Load indexes
        self.indexer.load_indexes()
        
        # Initialize retriever
        self.retriever = HybridRetriever(self.config, self.indexer, self.db)
        
        # Initialize chatbot
        self.chatbot = RAGChatbot(self.config, self.retriever)
        
        # Check Ollama connection
        if not self.chatbot.ollama.check_connection():
            print("⚠️  Warning: Cannot connect to Ollama!")
            print("   Make sure Ollama is running: ollama serve")
            print(f"   And the model is pulled: ollama pull {self.config.OLLAMA_MODEL}")
        
        print("✅ Chatbot initialized!")
    
    def chat(self, query: str) -> Dict:
        """Chat interface"""
        if not self.chatbot:
            raise RuntimeError("Chatbot not initialized. Call initialize_chatbot() first.")
        
        return self.chatbot.chat(query)
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return self.db.get_stats()


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Local RAG System for Mac Mini M4")
    parser.add_argument('--upload', type=str, help='PDF file to upload and index')
    parser.add_argument('--chat', action='store_true', help='Start interactive chat')
    parser.add_argument('--query', type=str, help='Single query')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--model', type=str, default='gemma2:2b', 
                       help='Ollama model to use (default: gemma2:2b)')
    parser.add_argument('--vision-model', type=str, default='gemma2:27b-vision',
                       help='Ollama vision model to use (default: gemma2:27b-vision)')
    
    args = parser.parse_args()
    
    # Initialize config
    config = LocalRAGConfig()
    config.OLLAMA_MODEL = args.model
    if hasattr(args, 'vision_model'):
        config.OLLAMA_VISION_MODEL = args.vision_model
    
    app = LocalRAGApplication(config)
    
    if args.upload:
        # Upload and index document
        app.index_document(args.upload)
    
    elif args.stats:
        # Show statistics
        stats = app.get_stats()
        print(f"\n{'='*70}")
        print("Database Statistics")
        print(f"{'='*70}")
        print(f"Documents: {stats['documents']}")
        print(f"Chunks: {stats['chunks']}")
        print(f"Avg chunk size: {stats['avg_chunk_size']} tokens")
        print(f"{'='*70}\n")
    
    elif args.query:
        # Single query
        app.initialize_chatbot()
        result = app.chat(args.query)
        print(f"\n{'='*70}")
        print("Response")
        print(f"{'='*70}")
        print(result['response'])
        print(f"\n{'='*70}")
        print("Sources")
        print(f"{'='*70}")
        for src in result['sources']:
            print(f"{src['source_id']}. {src['heading']} (score: {src['score']:.4f})")
        print(f"{'='*70}\n")
    
    elif args.chat:
        # Interactive chat
        app.initialize_chatbot()
        
        print(f"\n{'='*70}")
        print("RAG Chatbot - Interactive Mode")
        print(f"{'='*70}")
        print(f"Model: {config.OLLAMA_MODEL}")
        print(f"Documents: {app.get_stats()['documents']}")
        print(f"Chunks: {app.get_stats()['chunks']}")
        print("\nType 'exit' or 'quit' to end\n")
        
        while True:
            try:
                query = input("You: ")
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if not query.strip():
                    continue
                
                result = app.chat(query)
                print(f"\nAssistant: {result['response']}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
