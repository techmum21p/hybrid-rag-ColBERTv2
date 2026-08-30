"""
Streamlit RAG Chatbot with Image Understanding
==============================================

Complete Local RAG Chatbot - EXACT implementation from 00-doc-processor.ipynb
Features:
- Hybrid retrieval (BM25s + Jina ColBERT v2 + RRF + Reranking)
- Image extraction and analysis with Gemma3 vision model
- Markdown-aware semantic chunking
- Conversational interface with memory
- Terminal output showing elapsed times, chunks, and scores
"""

import os
# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import io
import time
import warnings
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Suppress deprecation warnings
warnings.filterwarnings('ignore', message='.*torch_dtype.*deprecated.*')

# Core libraries
import numpy as np
import torch
from PIL import Image as PILImage

# PDF and text processing
import pymupdf4llm
import fitz  # PyMuPDF
from transformers import AutoTokenizer

# Retrieval
import bm25s
from bm25s.hf import BM25HF
import Stemmer
from sentence_transformers import SentenceTransformer

# Database
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# LLM
import requests
import base64

# Streamlit
import streamlit as st


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RAGConfig:
    """Configuration for local RAG system"""
    base_dir: str = os.path.abspath(os.getcwd())
    db_path: str = None
    min_chunk_size: int = 256
    max_chunk_size: int = 512
    chunk_overlap: int = 128
    bm25_top_k: int = 100
    colbert_top_k: int = 100
    final_top_k: int = 15
    chat_model: str = "llama3.2:3b"
    vision_model: str = "gemma3:4b"
    embedding_model: str = "jinaai/jina-colbert-v2"
    ollama_url: str = "http://localhost:11434"
    ollama_timeout: int = 300
    bm25_index_path: str = None
    colbert_index_path: str = None
    images_dir: str = None
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"

    def __post_init__(self):
        if self.db_path is None:
            self.db_path = os.path.join(self.base_dir, "rag_local.db")
        if self.bm25_index_path is None:
            self.bm25_index_path = os.path.join(self.base_dir, "indexes", "bm25s")
        if self.colbert_index_path is None:
            self.colbert_index_path = os.path.join(self.base_dir, "indexes", "colbert")
        if self.images_dir is None:
            self.images_dir = os.path.join(self.base_dir, "extracted_images")


# ============================================================================
# DATABASE MODELS
# ============================================================================

class Base(DeclarativeBase):
    pass

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
    chunk_metadata = Column(Text)


# ============================================================================
# OLLAMA CLIENT WITH STREAMING SUPPORT
# ============================================================================

class OllamaClient:
    """Client for interacting with Ollama API with streaming support"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.base_url = config.ollama_url

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        images: List[str] = None,
        timeout: int = 300,
        stream: bool = False
    ) -> str:
        """Generate text with Ollama (with optional streaming)"""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }

        if system:
            payload["system"] = system

        if images:
            payload["images"] = images

        try:
            if stream:
                response = requests.post(url, json=payload, timeout=timeout, stream=True)
                response.raise_for_status()

                full_response = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            token = chunk["response"]
                            print(token, end='', flush=True)
                            full_response += token

                        if chunk.get("done", False):
                            break

                print()
                return full_response
            else:
                response = requests.post(url, json=payload, timeout=timeout)
                response.raise_for_status()
                return response.json()["response"]

        except requests.exceptions.Timeout:
            print(f"\n❌ Ollama timeout after {timeout}s")
            return ""
        except Exception as e:
            print(f"\n❌ Ollama error: {e}")
            return ""

    def analyze_image(self, image_path: str) -> Dict[str, str]:
        """Analyze image using Gemma3 multimodal model"""
        if not os.path.exists(image_path):
            return {
                'description': 'Image not found',
                'type': 'error',
                'ocr_text': ''
            }

        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            description_prompt = """Analyze this image carefully and provide detailed information:

1. TYPE: Classify this image (diagram, flowchart, chart, graph, table, screenshot, architecture diagram, code snippet, formula, etc.)

2. DESCRIPTION: Describe what the image shows in 2-3 detailed sentences. Include:
   - Main subject/purpose
   - Key components or elements
   - Relationships between elements (if applicable)
   - Colors, arrows, or visual indicators (if relevant)

3. TEXT: Extract ALL visible text from the image. This is CRITICAL for search accuracy.
   - Include labels, titles, legends, annotations
   - Include numbers, percentages, values
   - Include code, formulas, equations
   - Include any text in tables, boxes, or speech bubbles
   - Preserve the order and structure where possible
   - If no text is visible, write "No text visible"

Format your response EXACTLY as follows:
TYPE: [type]
DESCRIPTION: [description]
TEXT: [all extracted text]"""

            response = self.generate(
                model="gemma3:4b",
                prompt=description_prompt,
                images=[image_data],
                timeout=120,
                stream=False
            )

            result = {
                'description': 'No description generated',
                'type': 'unknown',
                'ocr_text': ''
            }

            if response:
                type_match = re.search(r'TYPE:\s*(.+)', response, re.IGNORECASE)
                if type_match:
                    result['type'] = type_match.group(1).strip().lower()

                desc_match = re.search(r'DESCRIPTION:([\s\S]*?)(?=TEXT:|$)', response, re.IGNORECASE)
                if desc_match:
                    result['description'] = desc_match.group(1).strip()

                text_match = re.search(r'TEXT:([\s\S]*)', response, re.IGNORECASE)
                if text_match:
                    ocr = text_match.group(1).strip()
                    if ocr.lower() != "no text visible":
                        result['ocr_text'] = ocr

            return result

        except Exception as e:
            print(f"❌ Error analyzing image {image_path}: {str(e)}")
            return {
                'description': f'Error analyzing image: {str(e)}',
                'type': 'error',
                'ocr_text': ''
            }

    def chat(
        self,
        messages: List[Dict[str, str]],
        context: str = None,
        stream: bool = True
    ) -> str:
        """Chat with context and streaming"""

        if context:
            system_msg = """You are a document question-answering assistant. Follow these rules with ABSOLUTE strictness:

!! CRITICAL RULES - NO EXCEPTIONS !!

1. You MUST ONLY use information explicitly stated in the context below
2. DO NOT use any knowledge outside the provided context
3. DO NOT make inferences, assumptions, or educated guesses
4. DO NOT mention products, services, or technologies not explicitly in the context
5. If information is NOT in the context, respond EXACTLY: "I don't have that information in the provided documents"
6. DO NOT provide links, URLs, or suggest where to find more information
7. DO NOT say things like "for the latest information" or "check the official website"
8. When answering, cite the specific source number (e.g., "According to Source 2...")

CONTEXT FROM DOCUMENTS:
""" + context + """

Remember: If it's not in the context above, you DON'T KNOW IT. Period."""
        else:
            system_msg = "You are a helpful AI assistant. Please provide accurate and helpful responses based only on what you know."

        prompt = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages
        ])

        return self.generate(
            model=self.config.chat_model,
            prompt=prompt,
            system=system_msg,
            timeout=self.config.ollama_timeout,
            stream=stream
        )


# ============================================================================
# MARKDOWN-AWARE SEMANTIC CHUNKER (EXACT FROM NOTEBOOK)
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
        return len(self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=512
        ))


# ============================================================================
# DOCUMENT PROCESSOR WITH IMAGE EXTRACTION (EXACT FROM NOTEBOOK)
# ============================================================================

class DocumentProcessor:
    """Handles PDF processing with image extraction and analysis"""

    def __init__(self, config: RAGConfig, ollama_client: OllamaClient):
        self.config = config
        self.ollama = ollama_client
        self.chunker = MarkdownSemanticChunker(config)
        os.makedirs(config.images_dir, exist_ok=True)

    def _sanitize_utf8(self, text: str) -> str:
        """Sanitize text to remove invalid UTF-8 characters"""
        if not text:
            return text

        try:
            return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"    ⚠️  UTF-8 sanitization error: {e}")
            return text.encode('ascii', errors='ignore').decode('ascii', errors='ignore')

    def pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert PDF to Markdown"""
        markdown_text = pymupdf4llm.to_markdown(pdf_path)
        return self._sanitize_utf8(markdown_text)

    def _group_nearby_rectangles(self, rects: List[fitz.Rect], proximity_threshold: float = 20) -> List[List[int]]:
        """Group rectangles that are close to each other"""
        if not rects:
            return []

        groups = []
        assigned = [False] * len(rects)

        for i, rect in enumerate(rects):
            if assigned[i]:
                continue

            current_group = [i]
            assigned[i] = True

            changed = True
            while changed:
                changed = False
                for j, other_rect in enumerate(rects):
                    if assigned[j]:
                        continue

                    for group_idx in current_group:
                        group_rect = rects[group_idx]

                        expanded_group = fitz.Rect(
                            group_rect.x0 - proximity_threshold,
                            group_rect.y0 - proximity_threshold,
                            group_rect.x1 + proximity_threshold,
                            group_rect.y1 + proximity_threshold
                        )

                        if expanded_group.intersects(other_rect):
                            current_group.append(j)
                            assigned[j] = True
                            changed = True
                            break

            groups.append(current_group)

        return groups

    def extract_images_from_pdf(
        self,
        pdf_path: str,
        document_id: int,
        min_image_size: int = 50,
        proximity_threshold: float = 20
    ) -> List[Dict]:
        """Extract images from PDF with intelligent grouping"""
        doc = fitz.open(pdf_path)
        images = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            if not image_list:
                continue

            image_bboxes = []
            for img_info in image_list:
                xref = img_info[0]
                rects = page.get_image_rects(xref)
                if rects:
                    for rect in rects:
                        width = rect.width
                        height = rect.height
                        if width >= min_image_size and height >= min_image_size:
                            image_bboxes.append({
                                'rect': rect,
                                'xref': xref,
                                'width': width,
                                'height': height
                            })

            if not image_bboxes:
                continue

            rects_only = [bbox['rect'] for bbox in image_bboxes]
            groups = self._group_nearby_rectangles(rects_only, proximity_threshold)

            for group_idx, group in enumerate(groups):
                if len(group) == 1:
                    bbox = image_bboxes[group[0]]
                    try:
                        base_image = doc.extract_image(bbox['xref'])
                        image_bytes = base_image["image"]
                        pil_image = PILImage.open(io.BytesIO(image_bytes))

                        image_filename = f"doc{document_id}_page{page_num+1}_img{len(images)+1}.png"
                        image_path = os.path.join(self.config.images_dir, image_filename)

                        if pil_image.mode == 'RGBA':
                            pil_image = pil_image.convert('RGB')

                        pil_image.save(image_path, 'PNG')

                        images.append({
                            'page_number': page_num + 1,
                            'image_path': image_path,
                            'image_index': len(images),
                            'is_composite': False,
                            'bbox': bbox['rect']
                        })
                    except Exception as e:
                        print(f"    ⚠️  Failed to extract single image on page {page_num+1}: {e}")

                else:
                    union_rect = image_bboxes[group[0]]['rect']
                    for idx in group[1:]:
                        union_rect = union_rect | image_bboxes[idx]['rect']

                    padding = 5
                    union_rect = fitz.Rect(
                        max(0, union_rect.x0 - padding),
                        max(0, union_rect.y0 - padding),
                        min(page.rect.width, union_rect.x1 + padding),
                        min(page.rect.height, union_rect.y1 + padding)
                    )

                    try:
                        mat = fitz.Matrix(2, 2)
                        pix = page.get_pixmap(matrix=mat, clip=union_rect)

                        img_data = pix.tobytes("png")
                        pil_image = PILImage.open(io.BytesIO(img_data))

                        image_filename = f"doc{document_id}_page{page_num+1}_composite{group_idx+1}.png"
                        image_path = os.path.join(self.config.images_dir, image_filename)

                        pil_image.save(image_path, 'PNG')

                        images.append({
                            'page_number': page_num + 1,
                            'image_path': image_path,
                            'image_index': len(images),
                            'is_composite': True,
                            'num_components': len(group),
                            'bbox': union_rect
                        })

                        print(f"    📊 Grouped {len(group)} images into composite on page {page_num+1}")

                    except Exception as e:
                        print(f"    ⚠️  Failed to create composite image on page {page_num+1}: {e}")

        doc.close()
        return images

    def analyze_images(
        self,
        images: List[Dict],
        document_id: int,
        db_session
    ) -> List[int]:
        """Analyze images with vision model"""
        image_ids = []

        for idx, img_info in enumerate(images):
            print(f"    Analyzing image {idx+1} on page {img_info['page_number']}...", end=' ')
            start_time = time.time()

            analysis = self.ollama.analyze_image(img_info['image_path'])

            image_record = Image(
                document_id=document_id,
                page_number=img_info['page_number'],
                image_path=img_info['image_path'],
                description=self._sanitize_utf8(analysis['description']),
                image_type=self._sanitize_utf8(analysis['type']),
                ocr_text=self._sanitize_utf8(analysis['ocr_text'])
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
        enriched_chunks = []

        for chunk in chunks:
            chunk_copy = chunk.copy()

            relevant_images = []

            for img in images_data:
                if any(keyword in chunk['text'].lower() for keyword in
                       ['figure', 'image', 'diagram', 'chart', 'screenshot', 'see below', 'shown in']):
                    relevant_images.append(img)

            if relevant_images:
                image_context = "\n\n[Images in this section]:\n"
                image_metadata = []

                for img in relevant_images:
                    image_context += f"- {img['type'].capitalize()}: {img['description']}\n"

                    if img.get('ocr_text') and img['ocr_text'].strip():
                        image_context += f"  Text visible in image: {img['ocr_text']}\n"

                    image_metadata.append({
                        'path': img['image_path'],
                        'description': img['description'],
                        'type': img['type'],
                        'ocr_text': img.get('ocr_text', '')
                    })

                chunk_copy['text'] = self._sanitize_utf8(chunk['text'] + image_context)
                chunk_copy['has_images'] = True
                chunk_copy['image_paths'] = [img['image_path'] for img in relevant_images]
                chunk_copy['image_metadata'] = image_metadata
            else:
                chunk_copy['text'] = self._sanitize_utf8(chunk['text'])
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

        # Step 3: Chunking
        print("\n[Step 3/5] Markdown-aware semantic chunking...", end=' ')
        start_time = time.time()
        doc_context = f"Document: {os.path.basename(pdf_path)}\n\n{markdown_text[:500]}"
        chunks = self.chunker.chunk_markdown(markdown_text, doc_context)
        elapsed = time.time() - start_time
        print(f"✓ {elapsed:.2f}s")
        print(f"  • Created {len(chunks)} semantic chunks")

        # Step 4: Enrich with images
        print("\n[Step 4/5] Enriching chunks with image context...", end=' ')
        start_time = time.time()
        if images_data:
            chunks = self.enrich_chunks_with_images(chunks, images_data, db_session)
            chunks_with_images = sum(1 for c in chunks if c.get('has_images', False))
            elapsed = time.time() - start_time
            print(f"✓ {elapsed:.2f}s")
            print(f"  • {chunks_with_images} chunks enriched with image context + OCR text")
        else:
            for chunk in chunks:
                chunk['text'] = self._sanitize_utf8(chunk['text'])
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
                text=self._sanitize_utf8(chunk['text']),
                heading_path=chunk.get('heading_path', ''),
                token_count=chunk.get('token_count', 0),
                has_images=chunk.get('has_images', False),
                chunk_metadata=json.dumps({
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
# JINA COLBERT V2 RETRIEVER (EXACT FROM NOTEBOOK)
# ============================================================================

class JinaColBERTRetriever:
    """Direct implementation of Jina ColBERT v2"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = SentenceTransformer(
            config.embedding_model,
            trust_remote_code=True,
            device=config.device
        )
        self.model.max_seq_length = 512
        self.corpus_embeddings = None
        self.corpus = None

    def index(self, corpus: List[str]) -> None:
        """Index corpus with ColBERT embeddings"""
        self.corpus = corpus

        print(f"  Encoding {len(corpus)} documents...")

        self.corpus_embeddings = self.model.encode(
            corpus,
            show_progress_bar=True,
            convert_to_tensor=True,
            batch_size=8
        )

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
        if not self.corpus or len(self.corpus) == 0:
            return []

        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True
        )

        scores = self._maxsim_score(query_embedding, self.corpus_embeddings)

        if len(self.corpus) == 1:
            return [{
                'document_id': 0,
                'score': float(scores.item() if scores.dim() == 0 else scores[0]),
                'text': self.corpus[0]
            }]

        k = min(k, len(scores))
        top_k_indices = torch.topk(scores, k=k).indices

        results = []
        for idx in top_k_indices:
            results.append({
                'document_id': int(idx),
                'score': float(scores[idx]),
                'text': self.corpus[idx] if self.corpus else None
            })

        return results

    def rerank(self, query: str, documents: List[str], k: int = 10) -> List[Dict]:
        """Rerank documents"""
        if not documents:
            return []

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        doc_embeddings = self.model.encode(
            documents,
            convert_to_tensor=True,
            batch_size=8
        )

        scores = self._maxsim_score(query_embedding, doc_embeddings)

        if len(documents) == 1:
            return [{
                'result_index': 0,
                'score': float(scores.item() if scores.dim() == 0 else scores[0]),
                'rank': 1,
                'text': documents[0]
            }]

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
        """Compute MaxSim score"""
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        if doc_embeddings.dim() == 1:
            doc_embeddings = doc_embeddings.unsqueeze(0)

        if query_embedding.dim() == 2 and doc_embeddings.dim() == 2:
            query_norm = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
            doc_norm = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)

            scores = torch.mm(query_norm, doc_norm.t())

            return scores.squeeze(0) if scores.size(0) == 1 else scores.squeeze()

        if query_embedding.dim() == 3:
            query_vec = query_embedding.mean(dim=1)
        else:
            query_vec = query_embedding

        if doc_embeddings.dim() == 3:
            doc_vec = doc_embeddings.mean(dim=1)
        else:
            doc_vec = doc_embeddings

        query_vec = torch.nn.functional.normalize(query_vec, p=2, dim=-1)
        doc_vec = torch.nn.functional.normalize(doc_vec, p=2, dim=-1)

        if query_vec.dim() == 1:
            query_vec = query_vec.unsqueeze(0)
        if doc_vec.dim() == 1:
            doc_vec = doc_vec.unsqueeze(0)

        scores = torch.mm(query_vec, doc_vec.t())

        return scores.squeeze(0) if scores.size(0) == 1 else scores.squeeze()


# ============================================================================
# DUAL INDEXER (EXACT FROM NOTEBOOK)
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

        stemmer = Stemmer.Stemmer("english")

        corpus_tokens = bm25s.tokenize(
            corpus,
            stopwords="en",
            stemmer=stemmer
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
# HYBRID RETRIEVER (EXACT FROM NOTEBOOK)
# ============================================================================

class HybridRetriever:
    """Three-stage retrieval: BM25s + ColBERT + Reranking"""

    def __init__(self, config: RAGConfig, indexer: DualIndexer, db_session, corpus_to_chunk_id: List[int] = None):
        self.config = config
        self.indexer = indexer
        self.db_session = db_session
        self.stemmer = Stemmer.Stemmer("english")
        self.corpus_to_chunk_id = corpus_to_chunk_id or []

    def retrieve(self, query: str, top_k_final: int = None) -> List[Dict]:
        """Three-stage hybrid retrieval"""
        if top_k_final is None:
            top_k_final = self.config.final_top_k

        print(f"\n🔍 Retrieving relevant chunks...")

        corpus_size = len(self.indexer.colbert_retriever.corpus) if self.indexer.colbert_retriever.corpus else 0

        bm25_k = min(self.config.bm25_top_k, corpus_size) if corpus_size > 0 else self.config.bm25_top_k
        colbert_k = min(self.config.colbert_top_k, corpus_size) if corpus_size > 0 else self.config.colbert_top_k

        print(f"   • Corpus size: {corpus_size}, using k={bm25_k} for retrieval")

        # Stage 1: BM25s
        start = time.time()
        bm25_results = self._bm25_search(query, k=bm25_k)
        bm25_time = time.time() - start
        print(f"   • BM25s: {bm25_time:.3f}s ({len(bm25_results)} results)")

        # Stage 2: ColBERT
        start = time.time()
        colbert_results = self._colbert_search(query, k=colbert_k)
        colbert_time = time.time() - start
        print(f"   • ColBERT: {colbert_time:.3f}s ({len(colbert_results)} results)")

        # Fusion
        start = time.time()
        fused_results = self._reciprocal_rank_fusion(bm25_results, colbert_results)
        candidates = fused_results[:min(50, len(fused_results))]
        fusion_time = time.time() - start
        print(f"   • Fusion: {fusion_time:.3f}s ({len(candidates)} candidates)")

        # Fetch chunks
        start = time.time()
        candidate_corpus_indices = [r['corpus_index'] for r in candidates]
        candidate_chunks = self._fetch_chunks_from_db(candidate_corpus_indices)

        # Preserve scores
        score_map = {}
        for bm25_result in bm25_results:
            idx = bm25_result['corpus_index']
            if idx not in score_map:
                score_map[idx] = {}
            score_map[idx]['bm25_score'] = bm25_result['score']

        for colbert_result in colbert_results:
            idx = colbert_result['corpus_index']
            if idx not in score_map:
                score_map[idx] = {}
            score_map[idx]['colbert_score'] = colbert_result['score']

        for fused_result in candidates:
            idx = fused_result['corpus_index']
            if idx in score_map:
                score_map[idx]['rrf_score'] = fused_result['rrf_score']

        for i, chunk in enumerate(candidate_chunks):
            corpus_idx = candidate_corpus_indices[i]
            if corpus_idx in score_map:
                chunk['intermediate_scores'] = score_map[corpus_idx]

        fetch_time = time.time() - start
        print(f"   • Fetch: {fetch_time:.3f}s ({len(candidate_chunks)} chunks)")

        # Stage 3: Rerank
        start = time.time()
        final_k = min(top_k_final, len(candidate_chunks))
        reranked_results = self._colbert_rerank(query, candidate_chunks, top_k=final_k)
        rerank_time = time.time() - start
        print(f"   • Rerank: {rerank_time:.3f}s (top {len(reranked_results)})")

        total_time = bm25_time + colbert_time + fusion_time + fetch_time + rerank_time
        print(f"   ✓ Total retrieval: {total_time:.3f}s")

        return reranked_results

    def _bm25_search(self, query: str, k: int) -> List[Dict]:
        """BM25s search"""
        query_tokens = bm25s.tokenize(
            query,
            stopwords="en",
            stemmer=self.stemmer
        )

        results, scores = self.indexer.bm25_retriever.retrieve(query_tokens, k=k)

        return [
            {'corpus_index': int(results[0][i]), 'score': float(scores[0][i]), 'source': 'bm25'}
            for i in range(len(results[0]))
        ]

    def _colbert_search(self, query: str, k: int) -> List[Dict]:
        """ColBERT search"""
        results = self.indexer.colbert_retriever.search(query=query, k=k)
        return [
            {'corpus_index': r['document_id'], 'score': r['score'], 'source': 'colbert'}
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
            corpus_idx = result['corpus_index']
            scores[corpus_idx] = scores.get(corpus_idx, 0) + (1 / (k + rank))

        for rank, result in enumerate(colbert_results, 1):
            corpus_idx = result['corpus_index']
            scores[corpus_idx] = scores.get(corpus_idx, 0) + (1 / (k + rank))

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{'corpus_index': idx, 'rrf_score': score} for idx, score in sorted_results]

    def _fetch_chunks_from_db(self, corpus_indices: List[int]) -> List[Dict]:
        """Fetch chunks from database"""
        chunks = []

        for corpus_idx in corpus_indices:
            if corpus_idx < len(self.corpus_to_chunk_id):
                chunk_id = self.corpus_to_chunk_id[corpus_idx]

                chunk = self.db_session.query(Chunk).filter_by(id=chunk_id).first()
                if chunk:
                    chunks.append({
                        'chunk_id': chunk.id,
                        'text': chunk.text,
                        'document_id': chunk.document_id,
                        'heading_path': chunk.heading_path,
                        'has_images': chunk.has_images,
                        'metadata': json.loads(chunk.chunk_metadata) if chunk.chunk_metadata else {}
                    })

        return chunks

    def _colbert_rerank(self, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
        """ColBERT reranking"""
        if not chunks:
            return []

        documents = [chunk['text'] for chunk in chunks]
        reranked_results = self.indexer.colbert_retriever.rerank(query=query, documents=documents, k=top_k)

        final_results = []
        for result in reranked_results:
            original_chunk = chunks[result['result_index']]
            intermediate_scores = original_chunk.get('intermediate_scores', {})

            final_results.append({
                'chunk_id': original_chunk['chunk_id'],
                'text': original_chunk['text'],
                'document_id': original_chunk['document_id'],
                'heading_path': original_chunk.get('heading_path', ''),
                'has_images': original_chunk.get('has_images', False),
                'metadata': original_chunk['metadata'],
                'score': result['score'],
                'rank': result['rank'],
                'bm25_score': intermediate_scores.get('bm25_score', 0.0),
                'colbert_score': intermediate_scores.get('colbert_score', 0.0),
                'rrf_score': intermediate_scores.get('rrf_score', 0.0)
            })
        return final_results


# ============================================================================
# RAG CHATBOT (EXACT FROM NOTEBOOK)
# ============================================================================

class RAGChatbot:
    """Complete RAG chatbot with streaming"""

    def __init__(self, config: RAGConfig, retriever: HybridRetriever, ollama_client: OllamaClient):
        self.config = config
        self.retriever = retriever
        self.ollama = ollama_client
        self.conversation_history = []

    def chat(self, query: str, stream: bool = True) -> Dict:
        """Process user query and generate response"""
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(query)

        # Build context
        context = self._build_context(retrieved_chunks)

        # Generate response
        if stream:
            print(f"\n🤖 Generating response (streaming)...\n")
        else:
            print(f"\n🤖 Generating response...", end=' ')

        start_time = time.time()

        self.conversation_history.append({
            'role': 'user',
            'content': query
        })

        response = self.ollama.chat(
            messages=self.conversation_history,
            context=context,
            stream=stream
        )

        elapsed = time.time() - start_time

        if not stream:
            print(f"✓ {elapsed:.1f}s")
        else:
            print(f"\n⏱️  Response generated in {elapsed:.1f}s")

        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })

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

            image_info = ""
            if chunk.get('has_images') and chunk.get('metadata', {}).get('image_paths'):
                num_images = len(chunk['metadata']['image_paths'])
                image_info = f" [Contains {num_images} image(s)]"

            context_parts.append(f"[Source {i}{heading}{image_info}]\n{chunk['text']}\n")

        return "\n".join(context_parts)

    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Format source citations"""
        sources = []

        for i, chunk in enumerate(chunks):
            source = {
                'source_id': i + 1,
                'chunk_id': chunk['chunk_id'],
                'document_id': chunk['document_id'],
                'heading': chunk.get('heading_path', ''),
                'score': chunk['score'],
                'bm25_score': chunk.get('bm25_score', 0.0),
                'colbert_score': chunk.get('colbert_score', 0.0),
                'rrf_score': chunk.get('rrf_score', 0.0),
                'has_images': chunk.get('has_images', False),
                'text': chunk['text'],
                'preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            }

            if chunk.get('has_images') and chunk.get('metadata'):
                image_paths = chunk['metadata'].get('image_paths', [])
                source['image_paths'] = image_paths

            sources.append(source)

        return sources

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


# ============================================================================
# RAG APPLICATION (EXACT FROM NOTEBOOK)
# ============================================================================

class RAGApplication:
    """Main application orchestrator"""

    def __init__(self, config: RAGConfig):
        self.config = config

        # Database setup
        # Ensure database directory exists
        db_dir = os.path.dirname(config.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        # SQLite URL with threading and timeout configuration for Streamlit
        db_url = f"sqlite:///{config.db_path}"
        self.engine = create_engine(
            db_url,
            connect_args={
                "check_same_thread": False,  # Required for Streamlit
                "timeout": 30  # 30 second timeout for database locks
            }
        )
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.db_session = Session()

        # Initialize components
        self.ollama = OllamaClient(config)
        self.processor = DocumentProcessor(config, self.ollama)
        self.indexer = DualIndexer(config)
        self.retriever = None
        self.chatbot = None
        self.corpus_to_chunk_id = []

    def check_ollama(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.config.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def check_duplicate_file(self, filename: str) -> Optional[Document]:
        """Check if a file with the same name already exists"""
        return self.db_session.query(Document).filter_by(filename=filename).first()

    def delete_document_and_data(self, document_id: int) -> None:
        """Delete a document and all its associated data"""
        print(f"\n🗑️  Removing existing document (ID: {document_id})...")

        # Delete chunks
        chunks = self.db_session.query(Chunk).filter_by(document_id=document_id).all()
        for chunk in chunks:
            self.db_session.delete(chunk)
        print(f"   • Deleted {len(chunks)} chunks")

        # Delete images and files
        images = self.db_session.query(Image).filter_by(document_id=document_id).all()
        for img in images:
            if os.path.exists(img.image_path):
                try:
                    os.remove(img.image_path)
                except:
                    pass
            self.db_session.delete(img)
        print(f"   • Deleted {len(images)} images")

        # Delete document
        doc = self.db_session.query(Document).filter_by(id=document_id).first()
        if doc:
            self.db_session.delete(doc)

        self.db_session.commit()
        print("   ✓ Document and associated data removed")

    def indexes_exist(self) -> bool:
        """Check if indexes exist"""
        bm25_exists = os.path.exists(os.path.join(self.config.bm25_index_path, "indices.npz"))
        colbert_exists = os.path.exists(os.path.join(self.config.colbert_index_path, "index.pt"))
        mapping_exists = os.path.exists(os.path.join(self.config.base_dir, "indexes", "corpus_mapping.pkl"))
        return bm25_exists and colbert_exists and mapping_exists

    def index_documents(self, pdf_paths: List[str], overwrite_duplicates: bool = True) -> None:
        """Index PDF documents - appends to existing database and rebuilds indexes"""

        if not self.check_ollama():
            print("❌ Ollama is not running!")
            return

        # Check for duplicates
        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)
            existing_doc = self.check_duplicate_file(filename)

            if existing_doc:
                if overwrite_duplicates:
                    print(f"\n⚠️  Duplicate file detected: {filename}")
                    self.delete_document_and_data(existing_doc.id)
                else:
                    print(f"\n⚠️  Skipping duplicate file: {filename}")
                    continue

        # Process new documents
        all_chunks = []
        for pdf_path in pdf_paths:
            chunks, doc_id = self.processor.process_document(pdf_path, self.db_session)
            all_chunks.extend(chunks)

        print(f"\n{'='*60}")
        print("Rebuilding Indexes with ALL Documents")
        print(f"{'='*60}")

        # Build corpus and mapping from ALL chunks in database
        all_db_chunks = self.db_session.query(Chunk).order_by(Chunk.id).all()
        corpus = []
        self.corpus_to_chunk_id = []

        for chunk in all_db_chunks:
            corpus.append(chunk.text)
            self.corpus_to_chunk_id.append(chunk.id)

        print(f"  • Total corpus: {len(corpus)} chunks (from all documents)")
        print(f"  • Chunk ID mapping: {len(self.corpus_to_chunk_id)} entries")

        # Build indexes (these will overwrite old indexes with complete data)
        self.indexer.build_bm25_index(corpus)
        self.indexer.build_colbert_index(corpus)

        # Save mapping
        mapping_path = os.path.join(self.config.base_dir, "indexes", "corpus_mapping.pkl")
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        with open(mapping_path, 'wb') as f:
            pickle.dump(self.corpus_to_chunk_id, f)

        print(f"\n✅ Documents indexed successfully!")

        # Get document stats
        doc_count = self.db_session.query(Document).count()
        print(f"   • Total documents in database: {doc_count}")

    def initialize_chatbot(self) -> bool:
        """Initialize chatbot with existing indexes"""

        if not self.check_ollama():
            print("❌ Ollama is not running!")
            return False

        # Check if indexes exist
        if not self.indexes_exist():
            print("⚠️  No indexes found. Please upload and index documents first.")
            return False

        print("Loading indexes...")
        try:
            self.indexer.load_indexes()
        except Exception as e:
            print(f"❌ Error loading indexes: {e}")
            print("  Please re-index your documents.")
            return False

        # Load mapping
        mapping_path = os.path.join(self.config.base_dir, "indexes", "corpus_mapping.pkl")
        try:
            with open(mapping_path, 'rb') as f:
                self.corpus_to_chunk_id = pickle.load(f)
            print(f"  • Loaded {len(self.corpus_to_chunk_id)} chunk ID mappings")
        except FileNotFoundError:
            print("  ⚠️  Warning: No corpus mapping found. Please re-index your documents.")
            self.corpus_to_chunk_id = []
            return False

        self.retriever = HybridRetriever(self.config, self.indexer, self.db_session, self.corpus_to_chunk_id)
        self.chatbot = RAGChatbot(self.config, self.retriever, self.ollama)

        print("✅ Chatbot initialized and ready!")
        return True

    def chat(self, query: str) -> Dict:
        """Chat interface"""
        if not self.chatbot:
            raise RuntimeError("Chatbot not initialized")

        return self.chatbot.chat(query)


# ============================================================================
# STREAMLIT UI
# ============================================================================

def init_session_state():
    """Initialize Streamlit session state"""
    if 'config' not in st.session_state:
        st.session_state.config = RAGConfig(chat_model='llama3.2:3b')

    if 'app' not in st.session_state:
        st.session_state.app = RAGApplication(st.session_state.config)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'sources' not in st.session_state:
        st.session_state.sources = []

    if 'chatbot_initialized' not in st.session_state:
        st.session_state.chatbot_initialized = False

    # Auto-initialize chatbot if indexes exist
    if 'auto_init_attempted' not in st.session_state:
        st.session_state.auto_init_attempted = False

    if not st.session_state.auto_init_attempted and st.session_state.app.indexes_exist():
        if st.session_state.app.check_ollama():
            success = st.session_state.app.initialize_chatbot()
            if success:
                st.session_state.chatbot_initialized = True
        st.session_state.auto_init_attempted = True


def render_sidebar():
    """Render sidebar with document upload and indexing"""
    with st.sidebar:
        st.title("📚 Document Management")

        # Check Ollama status
        ollama_running = st.session_state.app.check_ollama()

        if ollama_running:
            st.success("✅ Ollama is running")
        else:
            st.error("❌ Ollama is not running! Start it with: `ollama serve`")
            return

        st.divider()

        # File uploader
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])

        if uploaded_file is not None:
            # Check for duplicate
            existing_doc = st.session_state.app.check_duplicate_file(uploaded_file.name)

            if existing_doc:
                st.warning(f"⚠️ File '{uploaded_file.name}' already exists. It will be overwritten.")

            if st.button("📥 Index Document", type="primary"):
                # Save uploaded file temporarily
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("Processing document (this may take a few minutes)..."):
                    try:
                        st.session_state.app.index_documents([temp_path], overwrite_duplicates=True)
                        st.success("✅ Document indexed successfully!")

                        # Auto-initialize chatbot after indexing
                        with st.spinner("Initializing chatbot..."):
                            success = st.session_state.app.initialize_chatbot()
                            if success:
                                st.session_state.chatbot_initialized = True
                                st.session_state.auto_init_attempted = True  # Mark as initialized
                                st.success("✅ Chatbot ready!")
                            else:
                                st.error("❌ Failed to initialize chatbot")
                                st.info("💡 Try clicking '🤖 Initialize Chatbot' button below")

                    except Exception as e:
                        st.error(f"❌ Error processing document: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                # Only rerun if initialization was successful
                if st.session_state.chatbot_initialized:
                    st.rerun()
                else:
                    st.warning("⚠️ Document indexed but chatbot initialization failed. Check errors above.")

        st.divider()

        # Chatbot status and controls
        chatbot_object_exists = st.session_state.app.chatbot is not None

        if not st.session_state.chatbot_initialized or not chatbot_object_exists:
            # Show detailed status
            if not chatbot_object_exists:
                st.warning("⚠️ Chatbot object not initialized")

            # Manual initialization button (in case auto-init failed)
            if st.session_state.app.indexes_exist():
                if st.button("🤖 Initialize Chatbot"):
                    with st.spinner("Initializing chatbot..."):
                        success = st.session_state.app.initialize_chatbot()
                        if success:
                            st.session_state.chatbot_initialized = True
                            st.session_state.auto_init_attempted = True
                            st.success("✅ Chatbot ready!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to initialize. Check console for errors.")
            else:
                st.info("📚 Upload a document to get started!")
        else:
            st.success("✅ Chatbot is ready")

            if st.button("🗑️ Clear Conversation"):
                st.session_state.messages = []
                st.session_state.sources = []
                if st.session_state.app.chatbot:
                    st.session_state.app.chatbot.clear_history()
                st.rerun()

        st.divider()

        # Database stats
        st.subheader("📊 Database Stats")
        try:
            doc_count = st.session_state.app.db_session.query(Document).count()
            chunk_count = st.session_state.app.db_session.query(Chunk).count()
            image_count = st.session_state.app.db_session.query(Image).count()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Docs", doc_count)
            with col2:
                st.metric("Chunks", chunk_count)
            with col3:
                st.metric("Images", image_count)
        except Exception as e:
            st.warning(f"Unable to load database stats: {str(e)}")
            st.info("Try restarting the application if the issue persists.")


def render_chat():
    """Render chat interface"""
    st.title("💬 RAG Chatbot")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Verify chatbot is actually initialized (check both flag and object)
        if not st.session_state.chatbot_initialized or not st.session_state.app.chatbot:
            st.error("❌ Please initialize the chatbot first!")
            if st.session_state.app.indexes_exist():
                st.info("💡 Click the '🤖 Initialize Chatbot' button in the sidebar")
            else:
                st.info("💡 Upload a document first to get started")
            return

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.app.chat(prompt)
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
                    st.info("💡 Try reinitializing the chatbot or check the terminal for errors")
                    return
                st.markdown(result['response'])

        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": result['response']})
        st.session_state.sources = result['sources']

        st.rerun()


def render_sources():
    """Render retrieved sources"""
    if st.session_state.sources:
        st.divider()
        st.subheader(f"📊 Retrieved Sources ({len(st.session_state.sources)})")

        for src in st.session_state.sources:
            with st.expander(f"Source {src['source_id']} - {src['heading'] or 'No heading'}", expanded=False):
                # Display scores
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Final Score", f"{src['score']:.4f}")
                with col2:
                    st.metric("BM25", f"{src['bm25_score']:.4f}")
                with col3:
                    st.metric("ColBERT", f"{src['colbert_score']:.4f}")
                with col4:
                    st.metric("RRF", f"{src['rrf_score']:.4f}")

                # Display text
                st.text_area("Text", src['text'], height=200, disabled=True, key=f"text_{src['source_id']}")

                # Display images if available
                if src['has_images'] and src.get('image_paths'):
                    st.write("**Images:**")
                    for img_path in src['image_paths']:
                        if os.path.exists(img_path):
                            st.image(img_path, width=400)


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    init_session_state()

    # Layout
    render_sidebar()
    render_chat()
    render_sources()


if __name__ == "__main__":
    main()
