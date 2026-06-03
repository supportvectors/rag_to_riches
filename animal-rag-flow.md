# Animal quotes RAG flow

High-level flow for the **RAG to Riches** animal-quotes tutorial: index quotes once, then answer user questions with retrieval-augmented generation.

Original sketch: [`animal-rag-flow.png`](animal-rag-flow.png)

## Diagram

```mermaid
flowchart TB
    subgraph load["Phase 1 — One-time load"]
        quotes["Animal quotes\n(animals.jsonl)"]
        emb_load["Embedder\n(SimpleTextEmbedder)"]
        quotes --> emb_load
    end

    vdb[("VectorDB\n(Qdrant — animals collection)")]
    emb_load -->|"embed and index"| vdb

    subgraph realtime["Phase 2 — Real-time interaction"]
        uq["User query"]
        emb_query["Embedder"]
        retrieved["Top-K nearest results\n(quotes + metadata)"]
        sp["System prompt"]
        req["Request to LLM"]
        llm["LLM"]
        answer["Final response"]

        uq --> emb_query
        emb_query -->|"search"| vdb
        vdb -->|"top K nearest results"| retrieved
        uq --> req
        sp --> req
        retrieved --> req
        req --> llm --> answer
    end

    emb_load <-.->|"same model"| emb_query

    style load fill:#e8f5e9,stroke:#2e7d32
    style realtime fill:#e3f2fd,stroke:#1565c0
    style vdb fill:#fff8e1,stroke:#f9a825
    style llm fill:#fce4ec,stroke:#c2185b
    style answer fill:#f3e5f5,stroke:#7b1fa2
```

## Phase summary

| Phase | When | What happens |
|-------|------|----------------|
| **One-time load** | Setup / re-index | Each animal quote is embedded and stored in the vector database with metadata (author, category, etc.). |
| **Real-time interaction** | Each user question | The query is embedded, similar quotes are retrieved, then the LLM receives the system prompt, the user query, and the top-K context to produce the final answer. |

## Components in this repo

- **Corpus**: `data/corpus/animals/animals.jsonl` — loaded by [`Animals`](src/rag_to_riches/corpus/animals.py)
- **Embedder**: [`SimpleTextEmbedder`](src/rag_to_riches/vectordb/embedder.py)
- **Vector DB**: [`EmbeddedVectorDB`](src/rag_to_riches/vectordb/embedded_vectordb.py) (Qdrant)
- **End-to-end demo**: [`example_animals_usage.py`](src/examples/example_animals_usage.py) (index + search); full RAG via `Animals.ask_llm()` / `Animals.rag()`
