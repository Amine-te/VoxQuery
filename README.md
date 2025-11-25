# 🎤 VoxQuery - Voice to SQL

Transform your voice into executable SQL queries using a two-stage RAG (Retrieval-Augmented Generation) pipeline.

## 🏗️ Architecture

```
User Voice/Text Input
         ↓
    [Whisper ASR] ← Optional
         ↓
   Natural Language Query
         ↓
┌────────────────────────┐
│   RAG Engine Pipeline   │
├────────────────────────┤
│ Stage 1: Retrieve       │
│   Relevant Tables       │
│   (ChromaDB + Embeddings)│
├────────────────────────┤
│ Stage 2: Retrieve       │
│   Data Examples         │
│   (Filtered by Tables)  │
├────────────────────────┤
│ Prompt Construction     │
│   (Schema + Examples)   │
├────────────────────────┤
│ SQL Generation          │
│   (Ollama LLM)          │
└────────────────────────┘
         ↓
    Execute on MySQL
         ↓
    Return Results
```

## 📁 Project Structure

```
voxquery/
├── rag_engine.py          # Core RAG logic (reusable)
├── test.py                # CLI testing interface
├── app.py                 # Streamlit web interface
├── ingest_db.py           # Database indexing script
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── chroma_db/             # ChromaDB storage (generated)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo>
cd voxquery

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama (if not already installed)
# Visit: https://ollama.ai
```

### 2. Setup Database Indexing

```bash
# Configure your database in ingest_db.py
# Then run indexing
python ingest_db.py
```

This will create ChromaDB collections:
- `table_metadata` - Table schemas and relationships
- `table_data` - Sample data rows for context

### 3. Pull Ollama Model

```bash
# Pull your preferred model
ollama pull llama3:8b

# Or other models:
# ollama pull codellama:7b
# ollama pull mistral:7b
```

### 4. Run the Application

#### Option A: Streamlit Web Interface

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

#### Option B: Command Line

```bash
python test.py "Show me all trains departing from Casablanca"
```

## 🎯 Features

### Streamlit App Features
- ✅ **Multiple Input Methods**
  - 🎤 Live voice recording
  - 📁 Audio file upload
  - ⌨️ Direct text input

- ✅ **ASR Transcription**
  - Multiple Whisper model sizes
  - GPU acceleration
  - Real-time timing metrics

- ✅ **SQL Generation & Execution**
  - Two-stage RAG pipeline
  - Real-time performance metrics
  - Query result visualization

- ✅ **Comprehensive Debugging**
  - Stage 1: Retrieved table schemas
  - Stage 2: Data examples
  - Complete LLM prompt inspection
  - Timing breakdown

- ✅ **Full Configuration**
  - Database credentials
  - ASR model selection
  - LLM model selection
  - Retrieval parameters
  - GPU status monitoring

- ✅ **Modern UI**
  - VSCode-inspired dark theme
  - Responsive design
  - Intuitive controls

### CLI Test Script Features
- Simple command-line interface
- Verbose logging with emojis
- Context manager for cleanup
- Full pipeline testing

## 🔧 Configuration

### Database Settings
```python
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "your_password",
    "database": "your_database"
}
```

### RAG Engine Parameters
```python
rag_engine = RAGEngine(
    db_config=DB_CONFIG,
    ollama_model="llama3:8b",           # LLM model
    ollama_url="http://localhost:11434/api/generate",
    n_tables=5,                          # Tables to retrieve
    n_data_per_table=5,                  # Examples per table
    chroma_path="./chroma_db",           # Vector DB path
    embedding_model="all-MiniLM-L6-v2"   # Embedding model
)
```

### Whisper Models
Available models (size vs accuracy tradeoff):
- `tiny` - Fastest, least accurate
- `base` - Good balance (default)
- `small` - Better accuracy
- `medium` - High accuracy
- `large` - Best accuracy, slowest

## 📚 Usage Examples

### Using the RAG Engine Programmatically

```python
from rag_engine import RAGEngine

# Initialize
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "oncf_db"
}

# Use context manager for automatic cleanup
with RAGEngine(db_config, ollama_model="llama3:8b") as engine:
    result = engine.process_query(
        "Show me all trains arriving in Rabat today"
    )
    
    if result['success']:
        print(f"SQL: {result['sql_query']}")
        print(f"Rows: {result['results']['row_count']}")
        for row in result['results']['results']:
            print(row)
```

### Custom Retrieval Parameters

```python
# Override retrieval parameters per query
stage1 = engine.stage1_retrieve_tables(query, n_tables=10)
stage2 = engine.stage2_retrieve_data(query, table_names, n_data_per_table=8)
```

### Access Debug Information

```python
result = engine.process_query("your query")

# Check what was retrieved
print("Tables:", result['debug_info']['stage1_tables'])
print("Examples:", result['debug_info']['stage2_data'])
print("Prompt:", result['debug_info']['final_prompt'])

# Check timings
print("Timings:", result['timings'])
```

## 🐛 Debugging

### Enable Verbose Output
```python
result = engine.process_query("your query", verbose=True)
```

### Check GPU Status
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Test Ollama Connection
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3:8b",
  "prompt": "Hello"
}'
```

### Verify ChromaDB Collections
```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collections = client.list_collections()
print(collections)
```

## ⚡ Performance Tips

1. **Use GPU**: Ensure CUDA is available for 10-50x speedup
2. **Adjust retrieval**: More tables/examples = better context but slower
3. **Choose right Whisper model**: `base` is usually the sweet spot
4. **Ollama optimization**: Use quantized models for faster inference
5. **Index wisely**: Only index relevant sample data

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Support for more databases (PostgreSQL, SQLite)
- [ ] Query result caching
- [ ] Multi-turn conversations
- [ ] Query refinement based on errors
- [ ] Support for other LLM backends

## 📝 License

MIT License - Feel free to use in your projects!

## 🙏 Acknowledgments

- **Whisper** by OpenAI for ASR
- **ChromaDB** for vector storage
- **Ollama** for local LLM inference
- **Streamlit** for the web interface
- **Sentence Transformers** for embeddings

## 📧 Support

For issues or questions, please open a GitHub issue.

---