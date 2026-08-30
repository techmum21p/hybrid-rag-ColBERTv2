#!/bin/bash

# Launcher script for Streamlit RAG Chatbot
# This script checks if Ollama is running and starts the Streamlit app

echo "╔════════════════════════════════════════════════════════════╗"
echo "║          Streamlit RAG Chatbot Launcher                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Function to check if Ollama is running
check_ollama() {
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Check if Ollama is running
echo "🔍 Checking Ollama status..."
if check_ollama; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama is not running"
    echo ""
    echo "Please start Ollama in another terminal with:"
    echo "   ollama serve"
    echo ""
    echo "Or run this script with the --start-ollama flag:"
    echo "   ./start_rag.sh --start-ollama"
    echo ""

    # Check if user wants to start Ollama automatically
    if [[ "$1" == "--start-ollama" ]]; then
        echo "🚀 Starting Ollama in the background..."
        nohup ollama serve > /tmp/ollama.log 2>&1 &
        OLLAMA_PID=$!
        echo "   • Ollama PID: $OLLAMA_PID"
        echo "   • Log file: /tmp/ollama.log"

        # Wait for Ollama to start
        echo "⏳ Waiting for Ollama to be ready..."
        for i in {1..30}; do
            if check_ollama; then
                echo "✅ Ollama is ready!"
                break
            fi
            sleep 1
            echo -n "."
        done

        if ! check_ollama; then
            echo ""
            echo "❌ Ollama failed to start. Check /tmp/ollama.log for errors."
            exit 1
        fi
        echo ""
    else
        exit 1
    fi
fi

# Check if required models are available
echo ""
echo "🔍 Checking required Ollama models..."

# Function to check if model exists
check_model() {
    ollama list | grep -q "$1"
}

REQUIRED_MODELS=("llama3.2:3b" "gemma3:4b")
MISSING_MODELS=()

for model in "${REQUIRED_MODELS[@]}"; do
    if check_model "$model"; then
        echo "   ✅ $model"
    else
        echo "   ❌ $model (missing)"
        MISSING_MODELS+=("$model")
    fi
done

if [ ${#MISSING_MODELS[@]} -ne 0 ]; then
    echo ""
    echo "⚠️  Some required models are missing."
    echo "Install them with:"
    for model in "${MISSING_MODELS[@]}"; do
        echo "   ollama pull $model"
    done
    echo ""
    read -p "Do you want to continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if database and indexes exist
echo ""
echo "🔍 Checking for existing data..."
if [ -f "rag_local.db" ]; then
    echo "   ✅ Database found"
    DB_EXISTS=true
else
    echo "   ℹ️  No database found (will be created)"
    DB_EXISTS=false
fi

if [ -d "indexes/bm25s" ] && [ -d "indexes/colbert" ]; then
    echo "   ✅ Indexes found"
    INDEXES_EXIST=true
else
    echo "   ℹ️  No indexes found (upload documents to create)"
    INDEXES_EXIST=false
fi

# Show status summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    System Status                           ║"
echo "╠════════════════════════════════════════════════════════════╣"
if $DB_EXISTS && $INDEXES_EXIST; then
    echo "║  📚 Existing data found - chatbot will auto-initialize    ║"
elif $DB_EXISTS; then
    echo "║  📚 Database found but no indexes - re-upload documents   ║"
else
    echo "║  📚 No data found - upload documents to get started       ║"
fi
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Start Streamlit
echo "🚀 Starting Streamlit app..."
echo ""
echo "   The app will open in your browser at: http://localhost:8501"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

# Start Streamlit with the RAG app
streamlit run streamlit_rag.py

# Cleanup (only runs if user pressed Ctrl+C)
echo ""
echo "👋 Shutting down..."

# If we started Ollama, offer to stop it
if [[ "$1" == "--start-ollama" ]] && [ ! -z "$OLLAMA_PID" ]; then
    echo ""
    read -p "Do you want to stop Ollama? (y/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🛑 Stopping Ollama..."
        kill $OLLAMA_PID 2>/dev/null
        echo "✅ Ollama stopped"
    else
        echo "ℹ️  Ollama is still running (PID: $OLLAMA_PID)"
    fi
fi

echo ""
echo "✅ Cleanup complete. Goodbye!"
