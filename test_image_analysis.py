import os
import sys
from pathlib import Path

# Add parent directory to path to import local_rag_complete
sys.path.append(str(Path(__file__).parent))

from local_rag_complete import RAGConfig, OllamaClient

def test_image_analysis(image_path: str):
    # Create a minimal config
    config = RAGConfig(
        db_path=":memory:",
        bm25_index_path=":memory:",
        colbert_index_path=":memory:",
        images_dir=os.path.dirname(os.path.abspath(image_path)),
        ollama_url="http://localhost:11434",
        chat_model="gemma3:4b",
        ollama_timeout=300
    )
    
    # Initialize client
    client = OllamaClient(config)
    
    print(f"\n🖼️ Testing image analysis for: {image_path}")
    print("-" * 50)
    
    # Test image analysis
    result = client.analyze_image(image_path)
    
    print("\n📝 Analysis Results:")
    print(f"Type: {result['type']}")
    print(f"Description: {result['description']}")
    print(f"Extracted Text: {result['ocr_text']}")
    print("-" * 50)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_image_analysis.py <path_to_image>")
        sys.exit(1)
    
    image_path = os.path.abspath(sys.argv[1])
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    test_image_analysis(image_path)
