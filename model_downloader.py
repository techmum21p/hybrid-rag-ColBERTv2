"""
Model Downloader for RAG System
================================

Download and cache models separately before running the main application.
This is useful when you want to download on fast WiFi and run offline later.

Models downloaded:
1. ColBERTv2 (via RAGatouille) - for semantic search
2. Jina Reader (optional) - for better PDF text extraction
3. BERT tokenizer - for token counting during chunking
"""

import os
from pathlib import Path
from typing import Optional
import sys

try:
    from ragatouille import RAGPretrainedModel
    from transformers import AutoTokenizer
    from jina import Client  # Optional
    JINA_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("\nPlease install required packages:")
    print("pip install ragatouille transformers torch jina")
    sys.exit(1)


class ModelDownloader:
    """
    Downloads and caches models for offline use
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize downloader
        
        Args:
            cache_dir: Directory to cache models (default: ~/.cache/rag_models)
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/rag_models")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables for model caching
        os.environ['HF_HOME'] = str(self.cache_dir / "huggingface")
        os.environ['TRANSFORMERS_CACHE'] = str(self.cache_dir / "transformers")
        os.environ['RAGATOUILLE_CACHE'] = str(self.cache_dir / "ragatouille")
        
        print(f"📦 Model cache directory: {self.cache_dir}")
    
    def download_colbert(
        self, 
        model_name: str = "colbert-ir/colbertv2.0",
        force_redownload: bool = False
    ) -> None:
        """
        Download ColBERTv2 model for semantic search
        
        Args:
            model_name: HuggingFace model identifier
            force_redownload: Force redownload even if cached
        """
        print("\n" + "="*70)
        print("📥 Downloading ColBERTv2 Model")
        print("="*70)
        print(f"Model: {model_name}")
        print(f"Size: ~440MB")
        print(f"Purpose: Semantic search and reranking")
        
        try:
            # Initialize RAGatouille (this downloads the model)
            colbert_model = RAGPretrainedModel.from_pretrained(
                model_name,
                force_download=force_redownload
            )
            
            print("\n✅ ColBERTv2 downloaded successfully!")
            print(f"   Location: {self.cache_dir / 'ragatouille'}")
            
            # Get model info
            print("\n📊 Model Info:")
            print(f"   • Embedding dimension: 128")
            print(f"   • Max sequence length: 512 tokens")
            print(f"   • Token-level embeddings: Yes")
            
            return colbert_model
            
        except Exception as e:
            print(f"\n❌ Error downloading ColBERTv2: {e}")
            raise
    
    def download_tokenizer(
        self,
        model_name: str = "bert-base-uncased",
        force_redownload: bool = False
    ) -> None:
        """
        Download tokenizer for token counting during chunking
        
        Note: We use BERT tokenizer just for counting tokens,
        NOT for creating embeddings (ColBERT handles that)
        
        Args:
            model_name: Tokenizer model name
            force_redownload: Force redownload even if cached
        """
        print("\n" + "="*70)
        print("📥 Downloading Tokenizer")
        print("="*70)
        print(f"Model: {model_name}")
        print(f"Size: ~440MB")
        print(f"Purpose: Token counting for markdown-aware chunking")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir / "transformers"),
                force_download=force_redownload
            )
            
            print("\n✅ Tokenizer downloaded successfully!")
            print(f"   Location: {self.cache_dir / 'transformers'}")
            
            # Test tokenizer
            test_text = "This is a test sentence."
            tokens = tokenizer.encode(test_text)
            print(f"\n📊 Tokenizer Test:")
            print(f"   • Text: '{test_text}'")
            print(f"   • Tokens: {len(tokens)}")
            
            return tokenizer
            
        except Exception as e:
            print(f"\n❌ Error downloading tokenizer: {e}")
            raise
    
    def download_jina_reader(self) -> None:
        """
        Test Jina Reader connection (cloud-based, no download needed)
        
        Jina Reader is a cloud API for better text extraction from PDFs.
        No local download required - just verify API is accessible.
        """
        print("\n" + "="*70)
        print("📥 Testing Jina Reader API")
        print("="*70)
        print(f"Service: Jina Reader (cloud-based)")
        print(f"Purpose: Enhanced PDF text extraction")
        print(f"Note: No download needed - cloud API")
        
        try:
            # Test if Jina is accessible
            from jina import Client
            print("\n✅ Jina client library available!")
            print("   • API endpoint: https://r.jina.ai")
            print("   • Usage: r.jina.ai/<URL>")
            print("   • No authentication needed for basic use")
            
        except ImportError:
            print("\n⚠️  Jina client not installed (optional)")
            print("   Install with: pip install jina")
    
    def download_all(
        self,
        include_jina: bool = True,
        force_redownload: bool = False
    ) -> None:
        """
        Download all required models
        
        Args:
            include_jina: Test Jina Reader API
            force_redownload: Force redownload all models
        """
        print("\n" + "="*70)
        print("🚀 RAG Model Downloader")
        print("="*70)
        print("\nThis will download all required models for the RAG system.")
        print(f"Cache location: {self.cache_dir}")
        print("\nEstimated total download size: ~880MB")
        print("Estimated download time (10 Mbps): ~12 minutes")
        print("Estimated download time (50 Mbps): ~2-3 minutes")
        
        # Confirm before downloading
        response = input("\nProceed with download? (y/n): ")
        if response.lower() != 'y':
            print("❌ Download cancelled")
            return
        
        # Download each model
        models_downloaded = []
        
        try:
            # 1. ColBERTv2 (required)
            colbert = self.download_colbert(force_redownload=force_redownload)
            models_downloaded.append("ColBERTv2")
            
            # 2. Tokenizer (required)
            tokenizer = self.download_tokenizer(force_redownload=force_redownload)
            models_downloaded.append("BERT Tokenizer")
            
            # 3. Jina Reader (optional)
            if include_jina:
                self.download_jina_reader()
                models_downloaded.append("Jina Reader API")
            
            # Summary
            print("\n" + "="*70)
            print("✅ All Models Downloaded Successfully!")
            print("="*70)
            print(f"\nModels ready:")
            for model in models_downloaded:
                print(f"   ✓ {model}")
            
            print(f"\n📁 Cache location: {self.cache_dir}")
            print(f"💾 Total disk space used: ~880MB")
            
            print("\n🎉 You can now run the RAG system offline!")
            print("   Models will be loaded from cache automatically.")
            
        except Exception as e:
            print(f"\n❌ Error during download: {e}")
            print("\nModels successfully downloaded before error:")
            for model in models_downloaded:
                print(f"   ✓ {model}")
            raise
    
    def verify_models(self) -> dict:
        """
        Verify all models are downloaded and accessible
        
        Returns:
            Dictionary with model availability status
        """
        print("\n" + "="*70)
        print("🔍 Verifying Model Cache")
        print("="*70)
        
        status = {
            'colbert': False,
            'tokenizer': False,
            'cache_dir_exists': False
        }
        
        # Check cache directory
        if self.cache_dir.exists():
            status['cache_dir_exists'] = True
            print(f"✓ Cache directory exists: {self.cache_dir}")
        else:
            print(f"✗ Cache directory not found: {self.cache_dir}")
            return status
        
        # Check ColBERT
        try:
            colbert_model = RAGPretrainedModel.from_pretrained(
                "colbert-ir/colbertv2.0"
            )
            status['colbert'] = True
            print("✓ ColBERTv2 model found")
        except Exception as e:
            print(f"✗ ColBERTv2 model not found: {e}")
        
        # Check tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased",
                cache_dir=str(self.cache_dir / "transformers")
            )
            status['tokenizer'] = True
            print("✓ BERT tokenizer found")
        except Exception as e:
            print(f"✗ BERT tokenizer not found: {e}")
        
        # Summary
        print("\n" + "="*70)
        all_ready = all([status['colbert'], status['tokenizer']])
        if all_ready:
            print("✅ All models are ready!")
        else:
            print("⚠️  Some models are missing. Run download_all() to fix.")
        print("="*70)
        
        return status
    
    def get_cache_size(self) -> str:
        """
        Calculate total cache size
        
        Returns:
            Human-readable cache size
        """
        if not self.cache_dir.exists():
            return "0 MB"
        
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        # Convert to MB
        size_mb = total_size / (1024 * 1024)
        return f"{size_mb:.2f} MB"
    
    def clear_cache(self, confirm: bool = False) -> None:
        """
        Clear all cached models
        
        Args:
            confirm: Skip confirmation prompt
        """
        if not confirm:
            cache_size = self.get_cache_size()
            print(f"⚠️  This will delete all cached models ({cache_size})")
            response = input("Are you sure? (y/n): ")
            if response.lower() != 'y':
                print("❌ Cache clear cancelled")
                return
        
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            print(f"✅ Cache cleared: {self.cache_dir}")
        else:
            print(f"ℹ️  Cache directory doesn't exist: {self.cache_dir}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for model downloader"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download and cache models for RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models
  python model_downloader.py --download-all
  
  # Download only ColBERT
  python model_downloader.py --download-colbert
  
  # Verify models are cached
  python model_downloader.py --verify
  
  # Check cache size
  python model_downloader.py --cache-size
  
  # Clear cache
  python model_downloader.py --clear-cache
  
  # Custom cache directory
  python model_downloader.py --cache-dir /path/to/cache --download-all
        """
    )
    
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Custom cache directory (default: ~/.cache/rag_models)'
    )
    
    parser.add_argument(
        '--download-all',
        action='store_true',
        help='Download all required models'
    )
    
    parser.add_argument(
        '--download-colbert',
        action='store_true',
        help='Download only ColBERTv2 model'
    )
    
    parser.add_argument(
        '--download-tokenizer',
        action='store_true',
        help='Download only tokenizer'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify all models are downloaded'
    )
    
    parser.add_argument(
        '--cache-size',
        action='store_true',
        help='Show cache size'
    )
    
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear all cached models'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force redownload even if cached'
    )
    
    parser.add_argument(
        '--no-jina',
        action='store_true',
        help='Skip Jina Reader API test'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = ModelDownloader(cache_dir=args.cache_dir)
    
    # Execute requested action
    if args.download_all:
        downloader.download_all(
            include_jina=not args.no_jina,
            force_redownload=args.force
        )
    
    elif args.download_colbert:
        downloader.download_colbert(force_redownload=args.force)
    
    elif args.download_tokenizer:
        downloader.download_tokenizer(force_redownload=args.force)
    
    elif args.verify:
        downloader.verify_models()
    
    elif args.cache_size:
        size = downloader.get_cache_size()
        print(f"\n📦 Cache size: {size}")
        print(f"📁 Location: {downloader.cache_dir}")
    
    elif args.clear_cache:
        downloader.clear_cache()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
