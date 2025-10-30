"""
Markdown-Aware Semantic Chunking Strategy
==========================================

Better than token-based chunking because it:
- Preserves semantic boundaries (sections, subsections)
- Keeps related content together
- Handles both short and long sections intelligently
- Maintains document hierarchy for better context
"""

import re
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
from dataclasses import dataclass


@dataclass
class ChunkConfig:
    """Configuration for markdown-aware chunking"""
    min_chunk_size: int = 256   # Minimum tokens per chunk
    max_chunk_size: int = 1024  # Maximum tokens per chunk
    chunk_overlap: int = 128    # Overlap between chunks


class MarkdownSemanticChunker:
    """
    Intelligent markdown chunking that respects document structure
    
    Strategy:
    1. Parse markdown into hierarchical sections (H1 > H2 > H3)
    2. Create chunks at natural boundaries
    3. Merge small sections, split large ones
    4. Preserve heading context in each chunk
    """
    
    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def chunk_markdown(self, markdown_text: str) -> List[Dict]:
        """
        Main chunking method - creates semantically meaningful chunks
        
        Returns:
            List of chunks with metadata
        """
        # Step 1: Parse markdown into hierarchical sections
        sections = self._parse_markdown_hierarchy(markdown_text)
        
        # Step 2: Create chunks respecting boundaries
        chunks = self._create_chunks_from_sections(sections)
        
        # Step 3: Post-process (merge tiny, split huge)
        optimized_chunks = self._optimize_chunks(chunks)
        
        return optimized_chunks
    
    def _parse_markdown_hierarchy(self, text: str) -> List[Dict]:
        """
        Parse markdown into hierarchical sections
        
        Returns sections with:
        - level (1-6 for H1-H6)
        - title (heading text)
        - content (text under this heading)
        - parent_path (breadcrumb of parent headings)
        """
        lines = text.split('\n')
        sections = []
        current_section = None
        heading_stack = []  # Track hierarchy: [(level, title), ...]
        
        for line in lines:
            # Check if line is a heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if heading_match:
                # Save previous section if exists
                if current_section:
                    sections.append(current_section)
                
                # Parse new heading
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Update heading stack (remove deeper levels)
                heading_stack = [(lvl, ttl) for lvl, ttl in heading_stack if lvl < level]
                heading_stack.append((level, title))
                
                # Create parent path for context
                parent_path = ' > '.join([ttl for _, ttl in heading_stack[:-1]])
                
                # Start new section
                current_section = {
                    'level': level,
                    'title': title,
                    'content': '',
                    'parent_path': parent_path,
                    'full_path': ' > '.join([ttl for _, ttl in heading_stack])
                }
            else:
                # Add content to current section
                if current_section is not None:
                    current_section['content'] += line + '\n'
                else:
                    # Content before first heading
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
        
        # Add last section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _create_chunks_from_sections(self, sections: List[Dict]) -> List[Dict]:
        """
        Create chunks from sections, respecting semantic boundaries
        
        Strategy:
        - Each H1 section tries to be its own chunk
        - Small subsections (H2, H3) are grouped together
        - Large sections are split at paragraph boundaries
        """
        chunks = []
        current_chunk = None
        
        for section in sections:
            section_text = self._format_section_text(section)
            section_tokens = self._count_tokens(section_text)
            
            # Case 1: H1 or large section - try to make it standalone
            if section['level'] == 1 or section_tokens >= self.config.min_chunk_size:
                
                # Save previous chunk if exists
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = None
                
                # If section is too large, split it
                if section_tokens > self.config.max_chunk_size:
                    split_chunks = self._split_large_section(section)
                    chunks.extend(split_chunks)
                else:
                    # Section is good size, make it a chunk
                    chunks.append({
                        'text': section_text,
                        'metadata': {
                            'heading': section['full_path'],
                            'level': section['level'],
                            'token_count': section_tokens,
                            'type': 'section'
                        }
                    })
            
            # Case 2: Small subsection - accumulate with previous
            else:
                if current_chunk is None:
                    # Start new accumulation
                    current_chunk = {
                        'text': section_text,
                        'metadata': {
                            'heading': section['parent_path'] or section['title'],
                            'level': section['level'],
                            'token_count': section_tokens,
                            'type': 'accumulated',
                            'sections': [section['title']]
                        }
                    }
                else:
                    # Add to existing accumulation
                    combined_text = current_chunk['text'] + '\n\n' + section_text
                    combined_tokens = self._count_tokens(combined_text)
                    
                    if combined_tokens <= self.config.max_chunk_size:
                        # Can fit, add it
                        current_chunk['text'] = combined_text
                        current_chunk['metadata']['token_count'] = combined_tokens
                        current_chunk['metadata']['sections'].append(section['title'])
                    else:
                        # Too large, save current and start new
                        chunks.append(current_chunk)
                        current_chunk = {
                            'text': section_text,
                            'metadata': {
                                'heading': section['parent_path'] or section['title'],
                                'level': section['level'],
                                'token_count': section_tokens,
                                'type': 'accumulated',
                                'sections': [section['title']]
                            }
                        }
        
        # Add final accumulated chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_large_section(self, section: Dict) -> List[Dict]:
        """
        Split a large section at paragraph boundaries
        
        Preserves heading context in each split chunk
        """
        heading_text = f"# {section['title']}\n\n" if section['title'] else ""
        parent_context = f"Context: {section['parent_path']}\n\n" if section['parent_path'] else ""
        
        # Split content by paragraphs
        paragraphs = re.split(r'\n\n+', section['content'].strip())
        
        chunks = []
        current_text = heading_text + parent_context
        current_tokens = self._count_tokens(current_text)
        
        for para in paragraphs:
            para_tokens = self._count_tokens(para)
            
            if current_tokens + para_tokens <= self.config.max_chunk_size:
                # Add paragraph to current chunk
                current_text += para + '\n\n'
                current_tokens += para_tokens
            else:
                # Save current chunk and start new one
                if current_text.strip() != heading_text.strip():
                    chunks.append({
                        'text': current_text.strip(),
                        'metadata': {
                            'heading': section['full_path'],
                            'level': section['level'],
                            'token_count': current_tokens,
                            'type': 'split_section',
                            'part': len(chunks) + 1
                        }
                    })
                
                # Start new chunk with context
                current_text = heading_text + parent_context + para + '\n\n'
                current_tokens = self._count_tokens(current_text)
        
        # Add final chunk
        if current_text.strip():
            chunks.append({
                'text': current_text.strip(),
                'metadata': {
                    'heading': section['full_path'],
                    'level': section['level'],
                    'token_count': current_tokens,
                    'type': 'split_section',
                    'part': len(chunks) + 1
                }
            })
        
        return chunks
    
    def _optimize_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Post-process chunks:
        - Merge very small chunks
        - Add overlap between chunks for context continuity
        """
        optimized = []
        i = 0
        
        while i < len(chunks):
            chunk = chunks[i]
            
            # If chunk is too small and not the last one, try to merge
            if (chunk['metadata']['token_count'] < self.config.min_chunk_size and 
                i < len(chunks) - 1):
                
                next_chunk = chunks[i + 1]
                combined_text = chunk['text'] + '\n\n' + next_chunk['text']
                combined_tokens = self._count_tokens(combined_text)
                
                if combined_tokens <= self.config.max_chunk_size:
                    # Merge chunks
                    merged_chunk = {
                        'text': combined_text,
                        'metadata': {
                            'heading': chunk['metadata']['heading'],
                            'token_count': combined_tokens,
                            'type': 'merged',
                            'merged_sections': [
                                chunk['metadata'].get('heading'),
                                next_chunk['metadata'].get('heading')
                            ]
                        }
                    }
                    optimized.append(merged_chunk)
                    i += 2  # Skip next chunk as we merged it
                    continue
            
            # Add overlap with next chunk if applicable
            if i < len(chunks) - 1:
                chunk_with_overlap = self._add_overlap(chunk, chunks[i + 1])
                optimized.append(chunk_with_overlap)
            else:
                optimized.append(chunk)
            
            i += 1
        
        return optimized
    
    def _add_overlap(self, current_chunk: Dict, next_chunk: Dict) -> Dict:
        """
        Add overlap from next chunk to current chunk for continuity
        """
        overlap_tokens = self.config.chunk_overlap
        
        # Get last N tokens from current chunk and first M tokens from next
        next_text = next_chunk['text']
        next_tokens = self.tokenizer.encode(next_text, add_special_tokens=False)
        
        # Take first overlap_tokens from next chunk
        overlap_token_ids = next_tokens[:min(overlap_tokens, len(next_tokens))]
        overlap_text = self.tokenizer.decode(overlap_token_ids, skip_special_tokens=True)
        
        # Add to current chunk metadata (not to main text to avoid duplication in index)
        chunk_copy = current_chunk.copy()
        chunk_copy['metadata'] = current_chunk['metadata'].copy()
        chunk_copy['metadata']['overlap_preview'] = overlap_text[:200]
        
        return chunk_copy
    
    def _format_section_text(self, section: Dict) -> str:
        """Format section with heading and parent context"""
        parts = []
        
        # Add parent context for better retrieval
        if section['parent_path']:
            parts.append(f"[Document Context: {section['parent_path']}]")
        
        # Add heading
        if section['title'] and section['title'] != 'Introduction':
            heading_prefix = '#' * section['level']
            parts.append(f"{heading_prefix} {section['title']}")
        
        # Add content
        parts.append(section['content'].strip())
        
        return '\n\n'.join(parts)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_markdown_chunking():
    """Example showing the difference between strategies"""
    
    sample_markdown = """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn from data.

## Supervised Learning

Supervised learning uses labeled data to train models.

### Classification

Classification predicts discrete categories. Examples include spam detection and image recognition.

### Regression

Regression predicts continuous values like house prices or temperature.

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data.

### Clustering

Clustering groups similar data points together. K-means is a popular algorithm.

### Dimensionality Reduction

PCA and t-SNE reduce feature dimensions while preserving information.

# Deep Learning

Deep learning uses neural networks with multiple layers.

## Neural Network Architectures

### Convolutional Neural Networks (CNNs)

CNNs excel at image processing tasks. They use convolutional layers to detect features like edges and textures. Popular architectures include ResNet, VGG, and Inception. CNNs have revolutionized computer vision.

### Recurrent Neural Networks (RNNs)

RNNs process sequential data. They maintain hidden states to capture temporal dependencies. LSTMs and GRUs are popular variants that address vanishing gradient problems.

## Training Deep Networks

Training requires large datasets and computational resources. Techniques like batch normalization, dropout, and learning rate scheduling improve convergence.
"""
    
    # Initialize chunker
    config = ChunkConfig(
        min_chunk_size=100,
        max_chunk_size=300,
        chunk_overlap=50
    )
    chunker = MarkdownSemanticChunker(config)
    
    # Create chunks
    chunks = chunker.chunk_markdown(sample_markdown)
    
    # Display results
    print("="*70)
    print("MARKDOWN-AWARE SEMANTIC CHUNKING RESULTS")
    print("="*70)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n{'='*70}")
        print(f"CHUNK {i}")
        print(f"{'='*70}")
        print(f"Heading: {chunk['metadata']['heading']}")
        print(f"Type: {chunk['metadata']['type']}")
        print(f"Tokens: {chunk['metadata']['token_count']}")
        print(f"\nContent Preview:")
        print("-" * 70)
        print(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])
        print()
    
    print(f"\n{'='*70}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Average chunk size: {sum(c['metadata']['token_count'] for c in chunks) // len(chunks)} tokens")
    print("="*70)


# ============================================================================
# INTEGRATION WITH EXISTING RAG PIPELINE
# ============================================================================

def integrate_with_rag_pipeline():
    """
    Show how to integrate markdown-aware chunking into existing RAG pipeline
    """
    from markitdown import MarkItDown
    
    # Initialize components
    markitdown = MarkItDown()
    chunker = MarkdownSemanticChunker(
        ChunkConfig(
            min_chunk_size=256,
            max_chunk_size=1024,
            chunk_overlap=128
        )
    )
    
    # Process PDF
    pdf_path = "document.pdf"
    
    # Step 1: Convert to markdown (preserves structure)
    result = markitdown.convert(pdf_path)
    markdown_text = result.text_content
    
    # Step 2: Semantic chunking respecting markdown structure
    chunks = chunker.chunk_markdown(markdown_text)
    
    # Step 3: Each chunk now has rich metadata for contextualization
    for chunk in chunks:
        print(f"Heading: {chunk['metadata']['heading']}")
        print(f"Type: {chunk['metadata']['type']}")
        print(f"Size: {chunk['metadata']['token_count']} tokens")
        
        # This rich metadata can be used for:
        # - Better contextualization prompts to Gemini
        # - Improved retrieval (filter by section)
        # - Better citations (exact section references)


if __name__ == "__main__":
    print("Markdown-Aware Semantic Chunking Strategy\n")
    print("Benefits over token-based chunking:")
    print("âœ… Preserves document structure and hierarchy")
    print("âœ… Keeps semantically related content together")
    print("âœ… Handles short sections intelligently (merging)")
    print("âœ… Splits long sections at natural boundaries")
    print("âœ… Maintains heading context in each chunk")
    print("âœ… Better for tables, lists, code blocks\n")
    
    example_markdown_chunking()
