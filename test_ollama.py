#!/usr/bin/env python3
"""
Quick diagnostic script to test Ollama connectivity and model availability
"""

import requests
import json

OLLAMA_URL = "http://localhost:11434"

def check_ollama_running():
    """Check if Ollama service is running"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
        else:
            print(f"❌ Ollama returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama - is it running?")
        print("   Start it with: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def list_available_models():
    """List all available models in Ollama"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])

            if models:
                print(f"\n📦 Available models ({len(models)}):")
                for model in models:
                    name = model.get('name', 'unknown')
                    size = model.get('size', 0) / (1024**3)  # Convert to GB
                    print(f"   • {name} ({size:.2f} GB)")
            else:
                print("\n⚠️  No models found!")
                print("   Pull a model with: ollama pull llama3.2:3b")

            return [m.get('name') for m in models]
        else:
            print(f"❌ Failed to list models: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return []

def test_model(model_name):
    """Test if a specific model can generate text"""
    print(f"\n🧪 Testing model: {model_name}")
    print("   Sending simple prompt...")

    try:
        payload = {
            "model": model_name,
            "prompt": "Say 'Hello, I am working!' and nothing else.",
            "stream": False
        }

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=60  # 60 second timeout
        )

        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            print(f"   ✅ Model works! Response: {generated_text[:100]}")
            return True
        else:
            print(f"   ❌ Generation failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False

    except requests.exceptions.Timeout:
        print(f"   ❌ TIMEOUT - Model took too long to respond (>60s)")
        print(f"   This model may be too slow for your hardware")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("="*60)
    print("Ollama Diagnostic Tool")
    print("="*60)

    # Step 1: Check if Ollama is running
    if not check_ollama_running():
        return

    # Step 2: List available models
    available_models = list_available_models()

    if not available_models:
        return

    # Step 3: Test specific models
    print("\n" + "="*60)
    print("Testing Models")
    print("="*60)

    # Check for the models mentioned in the notebook
    models_to_test = ['gpt-oss:20b', 'llama3.2:3b', 'gemma3:4b', 'llava:7b']

    for model in models_to_test:
        # Check if model exists (with or without tag)
        model_exists = any(model in avail_model for avail_model in available_models)

        if model_exists:
            test_model(model)
        else:
            print(f"\n⚠️  Model '{model}' not found")
            print(f"   Pull it with: ollama pull {model}")

    # Recommendations
    print("\n" + "="*60)
    print("Recommendations")
    print("="*60)
    print("""
For Mac Mini M4 with 16GB RAM:
• ✅ RECOMMENDED: llama3.2:3b (fast, good quality)
• ✅ RECOMMENDED: gemma3:4b (multimodal, for images)
• ⚠️  SLOW: gpt-oss:20b (may be too large/slow)
• ⚠️  SLOW: llava:7b (okay for images, but slower)

If experiencing timeouts:
1. Use smaller models (3b-4b parameters)
2. Enable streaming to see progress
3. Increase timeout values
    """)

if __name__ == "__main__":
    main()
