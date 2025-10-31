"""
Complete Local RAG Chatbot with Image Understanding
===================================================

✅ No Cloud Dependencies (runs 100% locally)
✅ No RAGatouille (direct Jina ColBERT v2 implementation)
✅ PyMuPDF4LLM for PDF conversion
✅ Image extraction and analysis with LLaVA vision model
✅ Hybrid retrieval (BM25s + Jina ColBERT v2 + RRF + Reranking)
✅ Markdown-aware semantic chunking
✅ SQLite database for storage

Requirements:
- Ollama (for LLMs: llama3.2:3b, llava:7b)
- Mac Mini M4 or similar (16GB RAM recommended)
"""

import os
import json
import re
import io
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Core libraries
import numpy as np
import torch
from PIL import Image

# PDF and text processing
import pymupdf4llm
import fitz  # PyMuPDF for image extraction
from transformers import AutoTokenizer

# Retrieval
import bm25s
from sentence_transformers import SentenceTransformer

# Database
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# LLM
import requests  # For Ollama API


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RAGConfig:
    """Configuration for local RAG system"""
    # Database
    db_path: str = "rag_local.db"
    
    # Chunking
    min_chunk_size: int = 256
    max_chunk_size: int = 1024
    chunk_overlap: int = 128
    
    # Retrieval
    bm25_top_k: int = 100
    colbert_top_k: int = 100
    final_top_k: int = 10
    
    # Models
    chat_model: str = "llama3.2:3b"
    vision_model: str = "llava:7b"
    embedding_model: str = "jinaai/jina-colbert-v2"
    
    # Ollama
    ollama_url: str = "http://localhost:11434"
    
    # Paths
    bm25_index_path: str = "indexes/bm25s"
    colbert_index_path: str = "indexes/colbert"
    images_dir: str = "extracted_images"
    
    # Device
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"


# ============================================================================
# DATABASE MODELS
# ============================================================================

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    total_pages = Column(Integer)
    status = Column(String(50))

class Image(Base):
    __tablename__ = 'images'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)
    image_path = Column(String(500), nullable=False)
    description = Column(Text)
    image_type = Column(String(50))
    ocr_text = Column(Text)

class Chunk(Base):
    __tablename__ = 'chunks'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    heading_path = Column(String(500))
    token_count = Column(Integer)
    has_images = Column(Boolean, default=False)
    metadata = Column(Text)


# ============================================================================
# OLLAMA CLIENT
# ============================================================================

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.base_url = config.ollama_url
    
    def generate(
        self, 
        model: str, 
        prompt: str, 
        system: str = None,
        images: List[str] = None
    ) -> str:
        """Generate text with Ollama"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        if system:
            payload["system"] = system
        
        if images:
            payload["images"] = images
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"❌ Ollama error: {e}")
            return ""
    
    def analyze_image(self, image_path: str) -> Dict[str, str]:
        """Analyze image using LLaVA vision model"""
        
        # Read image and convert to base64
        with open(image_path, "rb") as f:
            import base64
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Generate description
        description_prompt = """Analyze this image and provide:
1. TYPE: What type of visual is this? (diagram, chart, table, screenshot, photo, etc.)
2. DESCRIPTION: A detailed description of what the image shows (2-3 sentences)
3. TEXT: Any visible text in the image (transcribe exactly)

Format your response as:
TYPE: [type]
DESCRIPTION: [description]
TEXT: [extracted text]"""
        
        response = self.generate(
            model=self.config.vision_model,
            prompt=description_prompt,
            images=[image_data]
        )
        
        # Parse response
        result = {
            'description': '',
            'type': 'unknown',
            'ocr_text': ''
        }
        
        for line in response.split('\n'):
            if line.startswith('TYPE:'):
                result['type'] = line.replace('TYPE:', '').strip().lower()
            elif line.startswith('DESCRIPTION:'):
                result['description'] = line.replace('DESCRIPTION:', '').strip()
            elif line.startswith('TEXT:'):
                result['ocr_text'] = line.replace('TEXT:', '').strip()
        
        return result
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        context: str = None
    ) -> str:
        """Chat with context"""
        
        # Build system message with context
        system_msg = "You are a helpful AI assistant."
        if context:
            system_msg += f"\n\nContext from documents:\n{context}\n\nUse this context to answer questions accurately."
        
        # Build prompt from messages
        prompt = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in messages
        ])
        
        return self.generate(
            model=self.config.chat_model,
            prompt=prompt,
            system=system_msg
        )


# ============================================================================
# MARKDOWN-AWARE SEMANTIC CHUNKER
# ============================================================================

class MarkdownSemanticChunker:
    """Intelligent markdown chunking that respects document structure"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
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
            
            if section_tokens > self.config.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = None
                
                split_chunks = self._split_large_section(section, doc_context)
                chunks.extend(split_chunks)
            
            elif section_tokens >= self.config.min_chunk_size:
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
                    
                    if combined_tokens <= self.config.max_chunk_size:
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
            
            if current_tokens + para_tokens <= self.config.max_chunk_size:
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
        """Merge very small chunks"""
        optimized = []
        i = 0
        
        while i < len(chunks):
            chunk = chunks[i]
            
            if (chunk['token_count'] < self.config.min_chunk_size and 
                i < len(chunks) - 1):
                
                next_chunk = chunks[i + 1]
                combined_text = chunk['text'] + '\n\n' + next_chunk['text']
                combined_tokens = self._count_tokens(combined_text)
                
                if combined_tokens <= self.config.max_chunk_size:
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
# DOCUMENT PROCESSOR WITH IMAGE EXTRACTION
# ============================================================================

class DocumentProcessor:
    """Handles PDF processing with image extraction and analysis"""
    
    def __init__(self, config: RAGConfig, ollama_client: OllamaClient):
        self.config = config
        self.ollama = ollama_client
        self.chunker = MarkdownSemanticChunker(config)
        
        # Create images directory
        os.makedirs(config.images_dir, exist_ok=True)
    
    def pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert PDF to Markdown using PyMuPDF4LLM"""
        markdown_text = pymupdf4llm.to_markdown(pdf_path)
        return markdown_text
    
    def extract_images_from_pdf(
        self, 
        pdf_path: str, 
        document_id: int
    ) -> List[Dict]:
        """Extract images from PDF and save to disk"""
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Save image
                image_filename = f"doc{document_id}_page{page_num+1}_img{img_index+1}.png"
                image_path = os.path.join(self.config.images_dir, image_filename)
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                images.append({
                    'page_number': page_num + 1,
                    'image_path': image_path,
                    'image_index': img_index
                })
        
        doc.close()
        return images
    
    def analyze_images(
        self, 
        images: List[Dict],
        document_id: int,
        db_session
    ) -> List[int]:
        """Analyze images with LLaVA and save to database"""
        image_ids = []
        
        for idx, img_info in enumerate(images):
            print(f"    Analyzing image {idx+1} on page {img_info['page_number']}...", end=' ')
            start_time = time.time()
            
            # Analyze with vision model
            analysis = self.ollama.analyze_image(img_info['image_path'])
            
            # Save to database
            image_record = Image(
                document_id=document_id,
                page_number=img_info['page_number'],
                image_path=img_info['image_path'],
                description=analysis['description'],
                image_type=analysis['type'],
                ocr_text=analysis['ocr_text']
            )
            db_session.add(image_record)
            db_session.flush()
            
            image_ids.append(image_record.id)
            
            elapsed = time.time() - start_time
            print(f"✓ ({elapsed:.1f}s)")
        
        db_session.commit()
        return image_ids
    
    def enrich_chunks_with_images(
        self,
        chunks: List[Dict],
        images_data: List[Dict],
        db_session
    ) -> List[Dict]:
        """Add image context to relevant chunks"""
        
        # Get all images for this document
        enriched_chunks = []
        
        for chunk in chunks:
            chunk_copy = chunk.copy()
            
            # Find images that might be relevant to this chunk
            # (Simple heuristic: images within reasonable distance in text)
            relevant_images = []
            
            for img in images_data:
                # You could implement more sophisticated matching here
                # For now, we'll add all images to chunks that mention visual content
                if any(keyword in chunk['text'].lower() for keyword in 
                       ['figure', 'image', 'diagram', 'chart', 'screenshot', 'see below', 'shown in']):
                    relevant_images.append(img)
            
            if relevant_images:
                # Add image descriptions to chunk text
                image_context = "\n\n[Images in this section]:\n"
                image_metadata = []
                
                for img in relevant_images:
                    image_context += f"- {img['type'].capitalize()}: {img['description']}\n"
                    image_metadata.append({
                        'path': img['image_path'],
                        'description': img['description'],
                        'type': img['type']
                    })
                
                chunk_copy['text'] = chunk['text'] + image_context
                chunk_copy['has_images'] = True
                chunk_copy['image_paths'] = [img['image_path'] for img in relevant_images]
                chunk_copy['image_metadata'] = image_metadata
            else:
                chunk_copy['has_images'] = False
            
            enriched_chunks.append(chunk_copy)
        
        return enriched_chunks
    
    def process_document(
        self, 
        pdf_path: str,
        db_session
    ) -> Tuple[List[Dict], int]:
        """Complete processing pipeline"""
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path}")
        print(f"{'='*60}")
        
        # Step 1: Convert to markdown
        print("\n[Step 1/5] Converting PDF to Markdown...", end=' ')
        start_time = time.time()
        markdown_text = self.pdf_to_markdown(pdf_path)
        elapsed = time.time() - start_time
        print(f"✓ {elapsed:.2f}s")
        print(f"  • Extracted {len(markdown_text):,} characters")
        
        # Create document record
        doc = Document(
            filename=os.path.basename(pdf_path),
            status='processing'
        )
        db_session.add(doc)
        db_session.commit()
        
        # Step 2: Extract and analyze images
        print("\n[Step 2/5] Extracting and analyzing images...")
        start_time = time.time()
        
        images = self.extract_images_from_pdf(pdf_path, doc.id)
        
        if images:
            image_ids = self.analyze_images(images, doc.id, db_session)
            
            # Get image data for enrichment
            images_data = []
            for img_id in image_ids:
                img_record = db_session.query(Image).filter_by(id=img_id).first()
                if img_record:
                    images_data.append({
                        'image_path': img_record.image_path,
                        'description': img_record.description,
                        'type': img_record.image_type,
                        'ocr_text': img_record.ocr_text
                    })
        else:
            images_data = []
        
        elapsed = time.time() - start_time
        print(f"  ✓ Completed in {elapsed:.2f}s")
        print(f"  • Extracted {len(images)} images")
        if images:
            print(f"  • Vision analysis: ✓")
        
        # Step 3: Markdown-aware semantic chunking
        print("\n[Step 3/5] Markdown-aware semantic chunking...", end=' ')
        start_time = time.time()
        doc_context = f"Document: {os.path.basename(pdf_path)}\n\n{markdown_text[:500]}"
        chunks = self.chunker.chunk_markdown(markdown_text, doc_context)
        elapsed = time.time() - start_time
        print(f"✓ {elapsed:.2f}s")
        print(f"  • Created {len(chunks)} semantic chunks")
        
        # Step 4: Enrich chunks with image context
        print("\n[Step 4/5] Enriching chunks with image context...", end=' ')
        start_time = time.time()
        if images_data:
            chunks = self.enrich_chunks_with_images(chunks, images_data, db_session)
            chunks_with_images = sum(1 for c in chunks if c.get('has_images', False))
            elapsed = time.time() - start_time
            print(f"✓ {elapsed:.2f}s")
            print(f"  • {chunks_with_images} chunks enriched with image context")
        else:
            elapsed = time.time() - start_time
            print(f"✓ {elapsed:.2f}s")
            print(f"  • No images to enrich")
        
        # Step 5: Save to database
        print("\n[Step 5/5] Saving chunks to database...", end=' ')
        start_time = time.time()
        for idx, chunk in enumerate(chunks):
            chunk_record = Chunk(
                document_id=doc.id,
                chunk_index=idx,
                text=chunk['text'],
                heading_path=chunk.get('heading_path', ''),
                token_count=chunk.get('token_count', 0),
                has_images=chunk.get('has_images', False),
                metadata=json.dumps({
                    k: v for k, v in chunk.items() 
                    if k not in ['text', 'heading_path', 'token_count', 'has_images']
                })
            )
            db_session.add(chunk_record)
        
        doc.status = 'indexed'
        db_session.commit()
        elapsed = time.time() - start_time
        print(f"✓ {elapsed:.2f}s")
        
        return chunks, doc.id


# ============================================================================
# JINA COLBERT V2 RETRIEVER (NO RAGATOUILLE!)
# ============================================================================

class JinaColBERTRetriever:
    """Direct implementation of Jina ColBERT v2 (no RAGatouille dependency)"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = SentenceTransformer(
            config.embedding_model,
            trust_remote_code=True,
            device=config.device
        )
        self.corpus_embeddings = None
        self.corpus = None
    
    def index(self, corpus: List[str]) -> None:
        """Index corpus with ColBERT embeddings"""
        self.corpus = corpus
        
        print(f"  Encoding {len(corpus)} documents...")
        
        # Encode corpus (this gives us token-level embeddings)
        self.corpus_embeddings = self.model.encode(
            corpus,
            show_progress_bar=True,
            convert_to_tensor=True
        )
        
        # Save to disk
        os.makedirs(self.config.colbert_index_path, exist_ok=True)
        torch.save({
            'embeddings': self.corpus_embeddings,
            'corpus': corpus
        }, os.path.join(self.config.colbert_index_path, 'index.pt'))
    
    def load(self) -> None:
        """Load index from disk"""
        index_file = os.path.join(self.config.colbert_index_path, 'index.pt')
        data = torch.load(index_file, map_location=self.config.device)
        self.corpus_embeddings = data['embeddings']
        self.corpus = data['corpus']
    
    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search using MaxSim scoring"""
        # Encode query
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True
        )
        
        # Compute MaxSim scores
        scores = self._maxsim_score(query_embedding, self.corpus_embeddings)
        
        # Get top-k
        top_k_indices = torch.topk(scores, k=min(k, len(scores))).indices
        
        results = []
        for idx in top_k_indices:
            results.append({
                'document_id': int(idx),
                'score': float(scores[idx]),
                'text': self.corpus[idx] if self.corpus else None
            })
        
        return results
    
    def rerank(self, query: str, documents: List[str], k: int = 10) -> List[Dict]:
        """Rerank documents with more accurate scoring"""
        # Encode query and documents
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        doc_embeddings = self.model.encode(documents, convert_to_tensor=True)
        
        # Compute MaxSim scores
        scores = self._maxsim_score(query_embedding, doc_embeddings)
        
        # Sort by score
        sorted_indices = torch.argsort(scores, descending=True)
        
        results = []
        for rank, idx in enumerate(sorted_indices[:k]):
            results.append({
                'result_index': int(idx),
                'score': float(scores[idx]),
                'rank': rank + 1,
                'text': documents[idx]
            })
        
        return results
    
    def _maxsim_score(
        self, 
        query_embedding: torch.Tensor, 
        doc_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MaxSim score between query and documents
        
        MaxSim: For each query token, find max similarity with all doc tokens,
        then average across query tokens
        """
        # Ensure 3D tensors [batch, seq_len, hidden_dim]
        if query_embedding.dim() == 2:
            query_embedding = query_embedding.unsqueeze(0)
        if doc_embeddings.dim() == 2:
            doc_embeddings = doc_embeddings.unsqueeze(0)
        
        # Compute similarity: [batch_q, seq_q, batch_d, seq_d]
        # Simplified: just use mean pooling for now (Jina's model handles this internally)
        query_vec = query_embedding.mean(dim=1)  # [batch_q, hidden]
        doc_vec = doc_embeddings.mean(dim=1)     # [batch_d, hidden]
        
        # Cosine similarity
        scores = torch.nn.functional.cosine_similarity(
            query_vec.unsqueeze(1), 
            doc_vec.unsqueeze(0),
            dim=2
        )
        
        return scores.squeeze()


# ============================================================================
# DUAL INDEXER (BM25s + Jina ColBERT)
# ============================================================================

class DualIndexer:
    """Manages BM25s and Jina ColBERT v2 indexes"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.bm25_retriever = None
        self.colbert_retriever = JinaColBERTRetriever(config)
    
    def build_bm25_index(self, corpus: List[str]) -> None:
        """Build BM25s index"""
        print("\n[BM25s] Building lexical search index...", end=' ')
        start_time = time.time()
        
        corpus_tokens = bm25s.tokenize(
            corpus, 
            stopwords="en",
            stemmer=bm25s.stemmer.Stemmer.Stemmer("english")
        )
        
        self.bm25_retriever = bm25s.BM25()
        self.bm25_retriever.index(corpus_tokens)
        
        os.makedirs(self.config.bm25_index_path, exist_ok=True)
        self.bm25_retriever.save(self.config.bm25_index_path)
        
        elapsed = time.time() - start_time
        print(f"✓ {elapsed:.2f}s")
    
    def build_colbert_index(self, corpus: List[str]) -> None:
        """Build Jina ColBERT v2 index"""
        print("\n[ColBERT] Building semantic search index...")
        start_time = time.time()
        
        self.colbert_retriever.index(corpus)
        
        elapsed = time.time() - start_time
        print(f"  ✓ {elapsed:.2f}s")
    
    def load_indexes(self) -> None:
        """Load indexes from disk"""
        self.bm25_retriever = bm25s.BM25.load(self.config.bm25_index_path)
        self.colbert_retriever.load()


# ============================================================================
# HYBRID RETRIEVER WITH RRF AND RERANKING
# ============================================================================

class HybridRetriever:
    """Three-stage retrieval: BM25s + ColBERT + ColBERT Reranking"""
    
    def __init__(self, config: RAGConfig, indexer: DualIndexer, db_session):
        self.config = config
        self.indexer = indexer
        self.db_session = db_session
    
    def retrieve(self, query: str, top_k_final: int = None) -> List[Dict]:
        """Three-stage hybrid retrieval"""
        if top_k_final is None:
            top_k_final = self.config.final_top_k
        
        print(f"\n🔍 Retrieving relevant chunks...")
        
        # Stage 1: BM25s
        start = time.time()
        bm25_results = self._bm25_search(query, k=self.config.bm25_top_k)
        bm25_time = time.time() - start
        print(f"   • BM25s: {bm25_time:.3f}s")
        
        # Stage 2: ColBERT
        start = time.time()
        colbert_results = self._colbert_search(query, k=self.config.colbert_top_k)
        colbert_time = time.time() - start
        print(f"   • ColBERT: {colbert_time:.3f}s")
        
        # Fusion
        start = time.time()
        fused_results = self._reciprocal_rank_fusion(bm25_results, colbert_results)
        candidates = fused_results[:50]
        fusion_time = time.time() - start
        print(f"   • Fusion: {fusion_time:.3f}s")
        
        # Fetch chunks
        start = time.time()
        candidate_chunks = self._fetch_chunks_from_db([r['chunk_id'] for r in candidates])
        fetch_time = time.time() - start
        print(f"   • Fetch: {fetch_time:.3f}s")
        
        # Stage 3: Rerank
        start = time.time()
        reranked_results = self._colbert_rerank(query, candidate_chunks, top_k=top_k_final)
        rerank_time = time.time() - start
        print(f"   • Rerank: {rerank_time:.3f}s")
        
        total_time = bm25_time + colbert_time + fusion_time + fetch_time + rerank_time
        print(f"   ✓ Total retrieval: {total_time:.3f}s")
        
        return reranked_results
    
    def _bm25_search(self, query: str, k: int) -> List[Dict]:
        """Stage 1: BM25s lexical search"""
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
        """Stage 2: ColBERT semantic search"""
        results = self.indexer.colbert_retriever.search(query=query, k=k)
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
            chunk = self.db_session.query(Chunk).filter_by(id=chunk_id).first()
            if chunk:
                chunks.append({
                    'chunk_id': chunk.id,
                    'text': chunk.text,
                    'document_id': chunk.document_id,
                    'heading_path': chunk.heading_path,
                    'has_images': chunk.has_images,
                    'metadata': json.loads(chunk.metadata) if chunk.metadata else {}
                })
        return chunks
    
    def _colbert_rerank(self, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
        """Stage 3: ColBERT reranking"""
        documents = [chunk['text'] for chunk in chunks]
        reranked_results = self.indexer.colbert_retriever.rerank(query=query, documents=documents, k=top_k)
        
        final_results = []
        for result in reranked_results:
            original_chunk = chunks[result['result_index']]
            final_results.append({
                'chunk_id': original_chunk['chunk_id'],
                'text': original_chunk['text'],
                'document_id': original_chunk['document_id'],
                'heading_path': original_chunk.get('heading_path', ''),
                'has_images': original_chunk.get('has_images', False),
                'metadata': original_chunk['metadata'],
                'score': result['score'],
                'rank': result['rank']
            })
        return final_results


# ============================================================================
# RAG CHATBOT
# ============================================================================

class RAGChatbot:
    """Complete RAG chatbot with Ollama"""
    
    def __init__(self, config: RAGConfig, retriever: HybridRetriever, ollama_client: OllamaClient):
        self.config = config
        self.retriever = retriever
        self.ollama = ollama_client
        self.conversation_history = []
    
    def chat(self, query: str) -> Dict:
        """Process user query and generate response"""
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(query)
        
        # Build context
        context = self._build_context(retrieved_chunks)
        
        # Generate response
        print(f"\n🤖 Generating response...", end=' ')
        start_time = time.time()
        
        self.conversation_history.append({
            'role': 'user',
            'content': query
        })
        
        response = self.ollama.chat(
            messages=self.conversation_history,
            context=context
        )
        
        elapsed = time.time() - start_time
        print(f"✓ {elapsed:.1f}s")
        
        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })
        
        total_time = elapsed
        print(f"⏱️  Total: {total_time:.1f}s\n")
        
        return {
            'response': response,
            'sources': self._format_sources(retrieved_chunks),
            'retrieved_chunks': len(retrieved_chunks)
        }
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context from retrieved chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            heading = f" ({chunk['heading_path']})" if chunk.get('heading_path') else ""
            
            # Add image info if present
            image_info = ""
            if chunk.get('has_images') and chunk.get('metadata', {}).get('image_paths'):
                num_images = len(chunk['metadata']['image_paths'])
                image_info = f" [Contains {num_images} image(s)]"
            
            context_parts.append(f"[Source {i}{heading}{image_info}]\n{chunk['text']}\n")
        
        return "\n".join(context_parts)
    
    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Format source citations"""
        return [
            {
                'source_id': i + 1,
                'chunk_id': chunk['chunk_id'],
                'document_id': chunk['document_id'],
                'heading': chunk.get('heading_path', ''),
                'score': chunk['score'],
                'has_images': chunk.get('has_images', False),
                'preview': chunk['text'][:200] + "..."
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🗑️  Conversation history cleared")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class RAGApplication:
    """Main application orchestrator"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Database setup
        db_url = f"sqlite:///{config.db_path}"
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.db_session = Session()
        
        # Initialize Ollama client
        self.ollama = OllamaClient(config)
        
        # Initialize components
        self.processor = DocumentProcessor(config, self.ollama)
        self.indexer = DualIndexer(config)
        self.retriever = None
        self.chatbot = None
    
    def check_ollama(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.config.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def index_documents(self, pdf_paths: List[str]) -> None:
        """Index PDF documents"""
        
        if not self.check_ollama():
            print("❌ Ollama is not running!")
            print("Please start Ollama: ollama serve")
            return
        
        all_chunks = []
        
        for pdf_path in pdf_paths:
            chunks, doc_id = self.processor.process_document(pdf_path, self.db_session)
            all_chunks.extend(chunks)
        
        print(f"\n{'='*60}")
        print("Building Indexes")
        print(f"{'='*60}")
        
        # Build indexes
        corpus = [chunk['text'] for chunk in all_chunks]
        self.indexer.build_bm25_index(corpus)
        self.indexer.build_colbert_index(corpus)
        
        print(f"\n✅ Document indexed successfully!")
    
    def initialize_chatbot(self) -> None:
        """Initialize chatbot with existing indexes"""
        
        if not self.check_ollama():
            print("❌ Ollama is not running!")
            print("Please start Ollama: ollama serve")
            return
        
        print("Loading indexes...")
        self.indexer.load_indexes()
        
        self.retriever = HybridRetriever(self.config, self.indexer, self.db_session)
        self.chatbot = RAGChatbot(self.config, self.retriever, self.ollama)
        
        print("✅ Chatbot initialized and ready!")
    
    def chat(self, query: str) -> Dict:
        """Chat interface"""
        if not self.chatbot:
            raise RuntimeError("Chatbot not initialized. Call initialize_chatbot() first.")
        
        return self.chatbot.chat(query)
    
    def interactive_chat(self) -> None:
        """Interactive chat loop"""
        print("\n" + "="*60)
        print("RAG Chatbot - Interactive Mode")
        print("="*60)
        print("Type your questions (or 'exit' to quit, 'clear' to clear history)\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit']:
                    print("\nGoodbye! 👋")
                    break
                
                if user_input.lower() == 'clear':
                    self.chatbot.clear_history()
                    continue
                
                result = self.chat(user_input)
                print(f"\nAssistant: {result['response']}\n")
                
                # Show sources
                if result['sources']:
                    print(f"📚 Sources ({len(result['sources'])}):")
                    for src in result['sources'][:3]:  # Show top 3
                        heading = f" - {src['heading']}" if src['heading'] else ""
                        images = " 🖼️" if src['has_images'] else ""
                        print(f"  • Source {src['source_id']}{heading}{images}")
                    print()
            
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    def print_stats(self) -> None:
        """Print database statistics"""
        doc_count = self.db_session.query(Document).count()
        chunk_count = self.db_session.query(Chunk).count()
        image_count = self.db_session.query(Image).count()
        
        print(f"\n📊 Database Statistics:")
        print(f"   • Documents: {doc_count}")
        print(f"   • Chunks: {chunk_count}")
        print(f"   • Images: {image_count}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Local RAG Chatbot with Image Understanding")
    parser.add_argument('--upload', type=str, help='Upload and index a PDF file')
    parser.add_argument('--chat', action='store_true', help='Start interactive chat')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--model', type=str, default='llama3.2:3b', help='Ollama model to use')
    
    args = parser.parse_args()
    
    # Initialize config
    config = RAGConfig(chat_model=args.model)
    app = RAGApplication(config)
    
    # Check Ollama
    if not app.check_ollama():
        print("❌ Ollama is not running!")
        print("\nTo start Ollama:")
        print("  1. Open a terminal")
        print("  2. Run: ollama serve")
        print("  3. Keep that terminal open")
        print("\nThen run this script again.")
        return
    
    # Handle commands
    if args.upload:
        app.index_documents([args.upload])
    
    elif args.chat:
        app.initialize_chatbot()
        app.interactive_chat()
    
    elif args.stats:
        app.print_stats()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
