# Rag to Riches

A simple tutorial introducing RAG (Retrieval Augmented Generation) to a broader audience.

This tutorial comprises a small corpus of animal quotes, over which we build an RAG application. In the process, we learn about:

- **Embedder** to convert the text to a vector
- **VectorDB (Qdrant)** to index the vectors for approximate k-nn search
- **Generator (LLM with a system prompt)** to create a coherent answer to the user's question with the given search results

## 📦 One-Time Load & Index Flow

Before you can query the RAG system, you need to load and index your quotes (one-time setup):

```
┌─────────────────────┐
│ Initialize Components│  ← VectorDB, Embedder, Animals class
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Load from JSONL     │  ← Read and validate quotes from file
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Generate Embeddings │  ← Convert each quote text to vector
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Index into Qdrant   │  ← Store vectors with metadata in DB
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Ready for Search    │  ← Quotes are now searchable
└─────────────────────┘
```

## 🎯 Real-Time RAG Query Flow

Once your quotes are indexed, you can query the RAG system in real-time:

```
┌─────────────┐
│  User Query │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Query Embedding │  ← Convert query to vector using embedder
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Vector Search   │  ← Search in Qdrant vector database
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Retrieve Quotes  │  ← Get top-k most relevant quotes
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Create Context  │  ← Format quotes + query into RAG context
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   Send to LLM    │  ← Generate answer using GPT model
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  LLM Response    │  ← Structured answer with insights
└──────────────────┘
```

### Detailed Flow Steps

1. **User Query**: User asks a natural language question (e.g., "What do animals teach us about friendship?")

2. **Query Embedding**: The query is converted to a vector representation using a sentence transformer model (`all-MiniLM-L6-v2`)

3. **Vector Search**: The embedded query is used to search the Qdrant vector database for semantically similar animal quotes using cosine similarity

4. **Retrieve Quotes**: Top-k most relevant quotes are retrieved with their similarity scores, metadata (author, category), and content

5. **Create RAG Context**: The retrieved quotes are formatted into a structured context that includes:
   - The user's query
   - System prompt with instructions
   - Formatted quotes with attribution
   - Instructions for the LLM

6. **Send to LLM**: The complete RAG context is sent to an LLM (GPT-4o or GPT-3.5-turbo) via OpenAI API

7. **LLM Response**: The LLM generates a structured response containing:
   - A comprehensive answer
   - Key insights
   - Relevant quotes used
   - Follow-up questions

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- `uv` package manager (or pip)
- OpenAI API key (optional, for LLM features)

### Installation

1. **Clone the repository** (if not already done):
   ```bash
   cd rag_to_riches
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Set up environment variables** (optional, for LLM features):
   ```bash
   cp sample.env .env
   # Edit .env and add your OPENAI_API_KEY
   export OPENAI_API_KEY="your-api-key-here"
   ```

4. **Verify the setup**:
   ```bash
   uv run python -c "from rag_to_riches import __version__; print('Setup complete!')"
   ```

## 📖 Demo Walkthrough

### Step 1: Choose Your Demo

There are several demo examples available:

- **Simple RAG Usage** (`simple_rag_usage.py`) - Minimal example showing the `rag()` facade method
- **RAG with Animals** (`rag_with_animals_example.py`) - Complete workflow demonstration
- **RAG Context Demo** (`rag_context_demo.py`) - Shows RAG context structure and LLM integration

### Step 2: Run the Simple Demo

Let's start with the simplest example:

```bash
uv run src/rag_to_riches/examples/simple_rag_usage.py
```

**What happens:**
1. Initializes `EmbeddedVectorDB` (connects to Qdrant)
2. Initializes `SimpleTextEmbedder` (loads sentence transformer model)
3. Creates `Animals` instance with the vector DB and embedder
4. Executes a query: "What do animals teach us about love?"
5. The `rag()` method automatically:
   - Searches for relevant quotes
   - Creates RAG context
   - Sends to LLM (if API key is set)
   - Returns structured response
6. Displays the results in a formatted table

### Step 3: Run the Complete Demo

For a more detailed walkthrough:

```bash
uv run src/rag_to_riches/examples/rag_with_animals_example.py
```

**This demo shows:**
1. **Component Initialization**: Vector DB and embedder setup
2. **Semantic Search**: Finding relevant quotes for a query
3. **Search Results Display**: Beautiful table showing matched quotes
4. **RAG Context Creation**: How quotes are formatted for LLM input
5. **One-step RAG**: Using `search_and_create_rag_context()` method
6. **Filtered Search**: Searching within specific categories

### Step 4: Run the Full LLM Integration Demo

For complete LLM integration (requires OpenAI API key):

```bash
uv run src/rag_to_riches/examples/rag_context_demo.py
```

**This demo demonstrates:**
1. Real vector search in the animals collection
2. RAG context generation with full structure
3. Simple LLM responses (plain text)
4. Structured LLM responses (with insights, quotes, follow-ups)
5. Filtered queries by category
6. Beautiful rich display formatting

### Step 5: Understanding the Output

Each demo will show:

- **Search Results**: A table with:
  - Quote content
  - Author
  - Category
  - Similarity score
  - Metadata

- **RAG Context**: The formatted prompt that would be sent to the LLM, including:
  - System instructions
  - User query
  - Retrieved quotes with attribution

- **LLM Response** (if API key is set): Structured output with:
  - Comprehensive answer
  - Key insights
  - Source quotes
  - Follow-up questions

## 💻 Code Example

Here's a minimal example of using the RAG system:

```python
from rag_to_riches.corpus.animals import Animals
from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB
from rag_to_riches.vectordb.embedder import SimpleTextEmbedder

# Initialize components
vector_db = EmbeddedVectorDB()
embedder = SimpleTextEmbedder()
animals = Animals(vector_db=vector_db, embedder=embedder)

# Simple search
results = animals.search("friendship with animals", limit=5)
animals.display_search_results(results)

# Complete RAG pipeline (one method call)
result = animals.rag(
    user_query="What do animals teach us about love?",
    limit=3,
    response_type="structured"
)

# Access results
print(result["llm_response"].answer)
print(f"Based on {len(result['search_results'])} quotes")
```

## 📚 Learning Path

1. **Start Simple**: Read `src/rag_to_riches/examples/simple_rag_usage.py` to understand the basic facade pattern

2. **Study the Core**: Examine `src/rag_to_riches/corpus/animals.py` - the `Animals` class contains the complete RAG implementation:
   - `search()` - Semantic search method
   - `create_rag_context()` - Context formatting
   - `ask_llm()` - LLM integration
   - `rag()` - Complete pipeline facade

3. **Explore Components**:
   - `EmbeddedVectorDB` - Vector database wrapper
   - `SimpleTextEmbedder` - Text embedding model
   - `SemanticSearch` - Search engine implementation

4. **Run Examples**: Execute all demo files to see different aspects of RAG

## 🎓 Tips

- Start by studying `animals.py` -- it contains the `Animals` class, which includes the entire RAG implementation
- The vector database (Qdrant) stores pre-indexed animal quotes - no need to index them yourself for the demos
- For LLM features, ensure your `OPENAI_API_KEY` is set in the environment
- Use `limit=5` for a good balance between quality and speed
- Try different queries to see how semantic search finds relevant quotes

## 📖 Additional Resources

- **Documentation**: See `docs/` directory for detailed API documentation
- **Notebooks**: Check `docs/notebooks/examples/` for Jupyter notebook tutorials
- **Course**: To learn RAG deeply, register for our course at https://supportvectors.ai/courses/rag-and-ai-search-bootcamp/

---

Remember, this is a basic outline of a complete RAG implementation. The demos show you how all the pieces fit together to create a working RAG system!

